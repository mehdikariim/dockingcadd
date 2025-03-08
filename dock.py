##############################
# dockcadd.py
##############################
import streamlit as st
import os
import re
import sys
import pandas as pd
import subprocess
from io import BytesIO
import zipfile

# Attempt to ensure py3Dmol is installed (only works if environment allows pip install at runtime)
try:
    import py3Dmol
except ImportError:
    st.warning("Installing py3Dmol library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py3Dmol"])
    import py3Dmol

# RDKit is also required. If not installed, it won't work. Attempt to install if missing.
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    st.error("RDKit not installed. Please install RDKit in your environment.")
    st.stop()

# PDBFixer, Biopython (for PDBList), etc.
try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    from Bio.PDB import PDBList
except ImportError:
    st.error("pdbfixer and/or biopython are not installed. Please install them in your environment.")
    st.stop()


##############################
# Helper Functions
##############################

def run_command_with_live_output(command, log_file):
    """
    Runs a command in a subprocess, capturing output to both console and a log file.
    Returns exit code.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    with open(log_file, 'w') as log:
        for line in process.stdout:
            # If you want logs in real-time on Streamlit, you could do:
            # st.write(line)
            log.write(line)
    return process.wait()


def keep_only_chain_A_with_fallback(pdb_id, out_dir="receptor_prep"):
    """
    1) Download the specified PDB.
    2) Attempt to extract chain A lines. If none found, fallback to entire PDB.
    3) Repair with PDBFixer and return final receptor path.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    pdbl = PDBList()
    raw_pdb = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=out_dir)
    raw_pdb_file = os.path.join(out_dir, f"{pdb_id}_raw.pdb")
    if os.path.exists(raw_pdb):
        os.rename(raw_pdb, raw_pdb_file)

    chainA_file = os.path.join(out_dir, f"{pdb_id}_chainA_only.pdb")
    chain_a_count = 0

    with open(raw_pdb_file, 'r') as infile, open(chainA_file, 'w') as outfile:
        for line in infile:
            if len(line) < 22:
                continue
            chain_id = line[21]

            if line.startswith(("ATOM", "HETATM")) and chain_id == 'A':
                outfile.write(line)
                chain_a_count += 1
            elif line.startswith("TER") and chain_id == 'A':
                outfile.write(line)
            elif line.startswith("END"):
                outfile.write(line)

    if chain_a_count == 0:
        file_to_fix = raw_pdb_file
    else:
        file_to_fix = chainA_file

    fixer = PDBFixer(filename=file_to_fix)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    
    final_receptor = os.path.join(out_dir, f"{pdb_id}_chainA_prepared.pdb")
    with open(final_receptor, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return final_receptor


def run_p2rank_and_get_center(pdb_file, pdb_id):
    """
    Run p2rank to predict pockets. Parse top pocket center (x, y, z).
    p2rank must be installed or accessible in p2rank_2.4.2/
    """
    cmd = ["p2rank_2.4.2/prank", "predict", "-f", pdb_file]
    log_file = f"p2rank_{pdb_id}.log"
    code = run_command_with_live_output(cmd, log_file)
    if code != 0:
        raise RuntimeError(f"p2rank prediction failed for {pdb_id}.")

    base_name = os.path.basename(pdb_file).replace(".pdb", "")
    csv_path = f"p2rank_2.4.2/test_output/predict_{base_name}/{base_name}.pdb_predictions.csv"

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    row = df.iloc[0]
    cx, cy, cz = float(row['center_x']), float(row['center_y']), float(row['center_z'])
    return (cx, cy, cz)


def convert_pdb_to_pdbqt_receptor(pdb_file, pdbqt_file):
    """
    Convert receptor PDB -> PDBQT using OpenBabel with flags for receptor
    """
    cmd = [
        "obabel",
        "-i", "pdb", pdb_file,
        "-o", "pdbqt",
        "-O", pdbqt_file,
        "-xr", "-xn", "-xp"
    ]
    subprocess.run(cmd, check=True)


def generate_multiple_conformers(mol, num_confs=3, minimize=True, useMMFF=True):
    """Generate multiple 3D conformers for an RDKit Mol."""
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
    if minimize:
        for cid in cids:
            if useMMFF and AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=200)
    return mol


def prepare_ligands(smiles_list=None, sdf_file=None, num_confs=3, out_dir="ligand_prep"):
    """
    Create PDB files for each conformer from either SMILES or an SDF.
    Return list of (pdb_path, label).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    results = []

    def write_confs_to_pdb(mol, base_name):
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        out_paths = []
        for i, cid in enumerate(conf_ids):
            tmp_mol = Chem.Mol(mol, False, cid)
            pdb_path = os.path.join(out_dir, f"{base_name}_conf{i+1}.pdb")
            Chem.MolToPDBFile(tmp_mol, pdb_path)
            out_paths.append(pdb_path)
        return out_paths

    # Option A: SMILES
    if smiles_list:
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                st.warning(f"Skipping invalid SMILES: {smi}")
                continue
            mol3d = generate_multiple_conformers(mol, num_confs=num_confs)
            base_name = f"lig_{idx+1}"
            conf_pdbs = write_confs_to_pdb(mol3d, base_name)
            for pdb_path in conf_pdbs:
                short_label = os.path.splitext(os.path.basename(pdb_path))[0]
                results.append((pdb_path, short_label))

    # Option B: SDF
    if sdf_file and os.path.isfile(sdf_file):
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        mol_count = 0
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            mol_count += 1
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"sdf_{mol_count}"
            mol3d = generate_multiple_conformers(mol, num_confs=num_confs)
            base_name = f"{name}_{mol_count}"
            conf_pdbs = write_confs_to_pdb(mol3d, base_name)
            for pdb_path in conf_pdbs:
                short_label = os.path.splitext(os.path.basename(pdb_path))[0]
                results.append((pdb_path, short_label))

    return results


def convert_pdb_to_pdbqt_ligand(input_pdb, output_pdbqt):
    """
    Convert ligand PDB -> PDBQT with OpenBabel
    """
    cmd = [
        "obabel",
        "-i", "pdb", input_pdb,
        "-o", "pdbqt",
        "-O", output_pdbqt,
        "-h"
    ]
    subprocess.run(cmd, check=True)


def perform_docking(
    smiles_list=None,
    sdf_file=None,
    pdb_id="5ZMA",
    num_confs=3,
    docking_folder="docking_results"
):
    """
    Master function that:
    1) Preps the receptor (keeps chain A, fallback if missing).
    2) Runs p2rank for binding site center.
    3) Converts receptor to PDBQT.
    4) Preps ligands -> PDB -> PDBQT.
    5) Runs vina docking for each ligand conformer.
    6) Merges final complexes and writes CSV of scores.
    Returns path to results CSV.
    """
    if not os.path.exists(docking_folder):
        os.makedirs(docking_folder, exist_ok=True)

    # 1) Receptor prep
    receptor_clean = keep_only_chain_A_with_fallback(pdb_id, out_dir=docking_folder)

    # 2) p2rank -> get center
    center_x, center_y, center_z = run_p2rank_and_get_center(receptor_clean, pdb_id)
    box_size = 20.0  # you can make this adjustable if you want

    receptor_pdbqt = os.path.join(docking_folder, f"{pdb_id}_chainA_prepared.pdbqt")
    convert_pdb_to_pdbqt_receptor(receptor_clean, receptor_pdbqt)

    # 3) Ligand prep
    ligands_folder = os.path.join(docking_folder, "ligands")
    prepared_ligands = prepare_ligands(
        smiles_list=smiles_list,
        sdf_file=sdf_file,
        num_confs=num_confs,
        out_dir=ligands_folder
    )

    results_csv = os.path.join(docking_folder, "docking_results.csv")
    with open(results_csv, 'w') as f:
        f.write("LigandFile,BestScore\n")

    # 4) Dock each conformer
    for (pdb_path, label) in prepared_ligands:
        ligand_pdbqt = os.path.join(docking_folder, f"{label}.pdbqt")
        convert_pdb_to_pdbqt_ligand(pdb_path, ligand_pdbqt)

        out_pdbqt = os.path.join(docking_folder, f"{label}_out.pdbqt")
        log_file = os.path.join(docking_folder, f"{label}_vina.log")

        # Attempt to call vina. You must have vina_1.2.5_linux_x86_64 in PATH or local folder
        vina_executable = "./vina_1.2.5_linux_x86_64"  # or "vina_1.2.5_linux_x86_64"
        vina_cmd = [
            vina_executable,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--out", out_pdbqt,
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(box_size),
            "--size_y", str(box_size),
            "--size_z", str(box_size),
            "--num_modes", "5"
        ]

        exit_code = run_command_with_live_output(vina_cmd, log_file)
        best_score = "N/A"

        if exit_code == 0:
            # Parse best score from log
            with open(log_file, 'r') as lf:
                for line in lf:
                    # For rank=1 line which typically starts with '   1    ...'
                    if re.match(r'^\s*1\s+', line):
                        parts = line.split()
                        if len(parts) >= 2:
                            best_score = parts[1]
                        break

            # Convert best pose -> PDB
            docked_ligand_pdb = os.path.join(docking_folder, f"{label}_docked.pdb")
            obabel_cmd = [
                "obabel",
                "-ipdbqt", out_pdbqt,
                "-opdb",
                "-O", docked_ligand_pdb,
                "-d"
            ]
            subprocess.run(obabel_cmd, check=True)

            # Merge receptor + ligand -> final complex
            final_complex = os.path.join(docking_folder, f"{label}_complex.pdb")
            with open(final_complex, 'w') as out_f:
                with open(receptor_clean, 'r') as rec_f:
                    for line in rec_f:
                        if line.startswith("END"):
                            continue
                        out_f.write(line)
                out_f.write("TER\n")
                with open(docked_ligand_pdb, 'r') as lig_f:
                    out_f.write(lig_f.read())
                out_f.write("END\n")

        with open(results_csv, 'a') as f:
            f.write(f"{label},{best_score}\n")

    return results_csv


##############################
# Streamlit App
##############################

def main():
    st.title("DockingCADD - Docking with p2rank & Vina")
    st.markdown(
        """
        **IMPORTANT**  
        This app requires external dependencies that may **not** be available on Streamlit Cloud:
        - AutoDock Vina 1.2.5  
        - Open Babel  
        - p2rank 2.4.2  
        - RDKit, pdbfixer, biopython, py3Dmol, etc.  

        If running on your own machine (or Docker container) where these are installed, the app should work.  
        On Streamlit Cloud, it will likely fail due to missing system-level installations.
        """
    )

    pdb_id_input = st.text_input("Enter PDB ID:", value="5ZMA")

    st.subheader("Ligand Input")
    input_mode = st.radio("Select input mode:", ["SMILES", "SDF"])

    smiles_list = []
    sdf_file = None

    if input_mode == "SMILES":
        smi_text = st.text_area("Enter one SMILES per line:")
        if smi_text.strip():
            smiles_list = [x.strip() for x in smi_text.splitlines() if x.strip()]
    else:
        sdf_file_upload = st.file_uploader("Upload SDF file:", type=["sdf"])
        if sdf_file_upload is not None:
            sdf_file = "uploaded_ligands.sdf"
            with open(sdf_file, "wb") as f:
                f.write(sdf_file_upload.getvalue())

    num_confs = st.number_input("Number of conformers per ligand:", min_value=1, max_value=20, value=3)

    if st.button("Run Docking"):
        if (not smiles_list) and (not sdf_file):
            st.error("Please provide SMILES or an SDF file.")
            return

        with st.spinner("Running docking..."):
            try:
                results_csv = perform_docking(
                    smiles_list=smiles_list,
                    sdf_file=sdf_file,
                    pdb_id=pdb_id_input,
                    num_confs=num_confs,
                    docking_folder="docking_results"
                )
                st.success("Docking complete!")
            except Exception as e:
                st.error(f"Docking failed: {e}")
                st.stop()

        # Show results
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            df['BestScore'] = pd.to_numeric(df['BestScore'], errors='coerce')
            st.write("## Docking Results")
            st.dataframe(df)

            # Show top complex
            df_sorted = df.dropna(subset=['BestScore']).sort_values(by='BestScore')
            if len(df_sorted) > 0:
                top_ligand = df_sorted.iloc[0]['LigandFile']
                top_score = df_sorted.iloc[0]['BestScore']
                st.write(f"**Best Docking Pose:** {top_ligand} with score {top_score}")

                top_complex_pdb = os.path.join("docking_results", f"{top_ligand}_complex.pdb")
                if os.path.exists(top_complex_pdb):
                    st.write("### 3D Visualization of Top Pose")
                    with open(top_complex_pdb, 'r') as f:
                        pdb_str = f.read()

                    viewer = py3Dmol.view(width=600, height=400)
                    viewer.addModel(pdb_str, 'pdb')
                    # Show receptor as cartoon
                    viewer.setStyle({'cartoon': {}})
                    # Show the ligand(s) as sticks
                    viewer.setStyle({'model': -1}, {'stick': {}})
                    viewer.zoomTo()
                    viewer.spin(False)
                    st.write(viewer.render(), unsafe_allow_html=True)

            # ZIP download
            st.write("### Download All Results")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for root, dirs, files in os.walk("docking_results"):
                    for file in files:
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, "docking_results")
                        zf.write(filepath, arcname)

            st.download_button(
                label="Download docking_results.zip",
                data=zip_buffer.getvalue(),
                file_name="docking_results.zip",
                mime="application/zip"
            )


if __name__ == "__main__":
    main()
