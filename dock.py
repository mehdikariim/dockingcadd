###########################
# dockcadd.py
###########################
import streamlit as st
import subprocess
import os
import sys
import re
import shutil
import py3Dmol
import zipfile
from io import BytesIO

# --- Try to install or confirm needed packages and binaries ---
def install_dependencies():
    """
    Attempt to install everything needed for the docking pipeline:
      - System packages (pymol, openbabel)
      - Python packages (rdkit-pypi, pdbfixer, biopython, etc.)
      - Autodock Vina 1.2.5
      - p2rank 2.4.2
    WARNING:
      This will likely fail on Streamlit Cloud or other restricted environments.
      Intended for local usage or Docker-based usage.
    """
    st.write("Attempting to install required system dependencies and Python libraries ...")
    st.write("This may take several minutes. Please watch the logs for any errors.")

    # 1) Install Python packages via pip
    # You could list them all in one command or separate them
    py_pkgs = [
        "rdkit-pypi", "biopython", "pdbfixer", "py3Dmol", "pandas", "requests",
        "openmm",  # optional for PDBFixer
    ]
    cmd_pip = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + py_pkgs
    try:
        subprocess.run(cmd_pip, check=True)
        st.write("Python libraries installed successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install Python packages. Error: {e}")
        return False

    # 2) System-level installations (apt-get)
    # Note that Streamlit Cloud typically won't allow this. 
    apt_packages = ["pymol", "openbabel"]
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + apt_packages, check=True)
        st.write("System packages installed successfully.")
    except Exception as e:
        st.warning(f"System install step failed or not permitted. {e}")

    # 3) AutoDock Vina 1.2.5
    # We'll attempt to download the binary if it's not present
    vina_bin = "vina_1.2.5_linux_x86_64"
    if not os.path.exists(vina_bin):
        st.write("Downloading AutoDock Vina 1.2.5...")
        vina_url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
        try:
            subprocess.run(["wget", "-q", vina_url, "-O", vina_bin], check=True)
            subprocess.run(["chmod", "+x", vina_bin], check=True)
            st.write("AutoDock Vina downloaded and made executable.")
        except Exception as e:
            st.error(f"Unable to download/install Vina. {e}")
            return False
    else:
        st.write("AutoDock Vina 1.2.5 binary already present.")

    # 4) p2rank 2.4.2
    # We'll attempt to download and extract if it's not present
    p2rank_dir = "p2rank_2.4.2"
    if not os.path.exists(p2rank_dir):
        st.write("Downloading p2rank 2.4.2 ...")
        p2rank_url = "https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz"
        tar_file = "p2rank_2.4.2.tar.gz"
        try:
            subprocess.run(["wget", "-q", p2rank_url, "-O", tar_file], check=True)
            subprocess.run(["tar", "-xzf", tar_file], check=True)
            st.write("p2rank successfully downloaded and extracted.")
        except Exception as e:
            st.error(f"Failed to download or extract p2rank. {e}")
            return False
    else:
        st.write("p2rank 2.4.2 folder already present.")

    return True

# --- Docking pipeline code below ---
import pandas as pd
import numpy as np

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# BioPython, PDBFixer
from Bio.PDB import PDBList
from pdbfixer import PDBFixer
from openmm.app import PDBFile


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
            # If you want to see real-time logs in the Streamlit UI, you could do:
            # st.write(line)
            log.write(line)
    return process.wait()


def keep_only_chain_A_with_fallback(pdb_id, out_dir="receptor_prep"):
    """
    1) Download PDB
    2) Attempt to extract chain A
    3) Fallback to entire PDB if chain A not present
    4) Repair with PDBFixer
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
    cmd = [
        "obabel",
        "-i", "pdb", pdb_file,
        "-o", "pdbqt",
        "-O", pdbqt_file,
        "-xr", "-xn", "-xp"
    ]
    subprocess.run(cmd, check=True)


def generate_multiple_conformers(mol, num_confs=3, minimize=True, useMMFF=True):
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
                continue
            mol3d = generate_multiple_conformers(mol, num_confs=num_confs)
            base_name = f"lig_{idx+1}"
            conf_pdbs = write_confs_to_pdb(mol3d, base_name)
            for pdb_path in conf_pdbs:
                short_label = os.path.splitext(os.path.basename(pdb_path))[0]
                results.append((pdb_path, short_label))

    # Option B: SDF
    if sdf_file and os.path.isfile(sdf_file):
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
    Main docking workflow:
      1) Prepare receptor (chain A or fallback).
      2) p2rank -> get best pocket center.
      3) Receptor to PDBQT.
      4) Prepare ligands -> PDBQT.
      5) Run Vina, parse best scores, write final complexes.
    Returns path to CSV with results.
    """
    if not os.path.exists(docking_folder):
        os.makedirs(docking_folder, exist_ok=True)

    receptor_clean = keep_only_chain_A_with_fallback(pdb_id, out_dir=docking_folder)
    center_x, center_y, center_z = run_p2rank_and_get_center(receptor_clean, pdb_id)
    box_size = 20.0

    receptor_pdbqt = os.path.join(docking_folder, f"{pdb_id}_chainA_prepared.pdbqt")
    convert_pdb_to_pdbqt_receptor(receptor_clean, receptor_pdbqt)

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

    # Path to vina binary (assuming it's in the current directory)
    vina_bin = os.path.join(".", "vina_1.2.5_linux_x86_64")

    for (pdb_path, label) in prepared_ligands:
        ligand_pdbqt = os.path.join(docking_folder, f"{label}.pdbqt")
        convert_pdb_to_pdbqt_ligand(pdb_path, ligand_pdbqt)

        out_pdbqt = os.path.join(docking_folder, f"{label}_out.pdbqt")
        log_file = os.path.join(docking_folder, f"{label}_vina.log")

        vina_cmd = [
            vina_bin,
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
            with open(log_file, 'r') as lf:
                for line in lf:
                    # Look for line that starts with rank=1
                    if re.match(r'^\s*1\s+', line):
                        parts = line.split()
                        if len(parts) >= 2:
                            best_score = parts[1]
                        break

            # Convert best pose to PDB
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


# =============== Streamlit UI ===============

def main():
    st.title("DockCADD: Docking Workflow with p2rank & AutoDock Vina")

    # Attempt to install everything on script startup
    # (not typical for Streamlit Cloud, but okay for local usage).
    if "installed" not in st.session_state:
        st.session_state["installed"] = False
    if not st.session_state["installed"]:
        with st.spinner("Installing dependencies..."):
            success = install_dependencies()
        st.session_state["installed"] = success
        if not success:
            st.warning("Some installations failed. You may need to run this locally or in Docker.")
        else:
            st.success("All dependencies are (hopefully) installed!")
        st.stop()

    st.markdown(
        """
        **Instructions**:
        1. Provide a PDB ID (e.g. `5ZMA`).
        2. Provide ligands as SMILES (one per line) or upload an SDF.
        3. Choose # of conformers per ligand.
        4. Click **Run Docking**.
        
        **Note**: This app attempts to install everything at runtime, but typically
        you must run it in a local or Docker environment that allows system package installation.
        """
    )

    pdb_id_input = st.text_input("Enter PDB ID", value="5ZMA")

    st.write("### Ligand Input")
    input_mode = st.radio("Select Input Mode:", ["SMILES", "SDF"])

    smiles_list = []
    sdf_file_path = None

    if input_mode == "SMILES":
        smi_text = st.text_area("Enter one SMILES per line:")
        if smi_text.strip():
            smiles_list = [x.strip() for x in smi_text.splitlines() if x.strip()]
    else:
        # SDF file
        sdf_upload = st.file_uploader("Upload SDF file", type=["sdf"])
        if sdf_upload is not None:
            sdf_file_path = "uploaded_ligands.sdf"
            with open(sdf_file_path, "wb") as f:
                f.write(sdf_upload.getvalue())

    num_confs = st.number_input("Number of conformers per ligand", min_value=1, max_value=20, value=3)

    if st.button("Run Docking"):
        if not pdb_id_input.strip():
            st.error("Please provide a PDB ID.")
            st.stop()

        with st.spinner("Running docking workflow..."):
            results_csv_path = perform_docking(
                smiles_list=smiles_list if input_mode == "SMILES" else None,
                sdf_file=sdf_file_path if input_mode == "SDF" else None,
                pdb_id=pdb_id_input.strip(),
                num_confs=num_confs,
                docking_folder="docking_results"
            )

        st.success("Docking complete!")
        
        # Display results
        if os.path.exists(results_csv_path):
            df = pd.read_csv(results_csv_path)
            # Convert BestScore to numeric if possible
            df["BestScore"] = pd.to_numeric(df["BestScore"], errors='coerce')
            st.dataframe(df)

            df_sorted = df.dropna(subset=["BestScore"]).sort_values(by="BestScore")
            if len(df_sorted) > 0:
                top_ligand = df_sorted.iloc[0]["LigandFile"]
                top_score = df_sorted.iloc[0]["BestScore"]
                st.write(f"**Best Docking Pose**: {top_ligand} with score {top_score} kcal/mol")

                # Show 3D with py3Dmol
                top_complex_pdb = os.path.join("docking_results", f"{top_ligand}_complex.pdb")
                if os.path.exists(top_complex_pdb):
                    st.subheader("Top Complex Visualization (py3Dmol)")
                    with open(top_complex_pdb, 'r') as f:
                        pdb_str = f.read()

                    viewer = py3Dmol.view(width=600, height=400)
                    viewer.addModel(pdb_str, 'pdb')
                    # Receptor as cartoon
                    viewer.setStyle({'chain': 'A'}, {"cartoon": {}})
                    # Ligand as stick
                    viewer.setStyle({'model': -1}, {"stick": {}})
                    viewer.zoomTo()
                    viewer.spin(False)
                    st.write(viewer.render(), unsafe_allow_html=True)

        # Let user download a ZIP of results
        st.subheader("Download All Results")
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for root, dirs, files in os.walk("docking_results"):
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp, arcname=os.path.relpath(fp, "docking_results"))

        st.download_button(
            label="Download docking_results.zip",
            data=zip_buffer.getvalue(),
            file_name="docking_results.zip",
            mime="application/zip"
        )


if __name__ == "__main__":
    main()
