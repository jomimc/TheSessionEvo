"""Protein structural analysis: RSA from AlphaFold structures and Cα distance matrices."""

import csv
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

import numpy as np

from .utils import THREE_TO_ONE, MAX_ASA, parse_fasta, download_alphafold_structure


def extract_uniprot_accession(blast_accession):
    """
    Extract a clean UniProt accession from a BLAST hit accession string.
    Handles formats like P00004, sp|P00004|CYC_HORSE, sp|P00004.2|CYC_HORSE.
    Strips version suffixes (.2, .3) not accepted by AlphaFold DB.
    """
    acc = blast_accession.strip()
    if "|" in acc:
        parts = acc.split("|")
        if len(parts) >= 2:
            acc = parts[1]
    return acc.split(".")[0]


def compute_rsa_from_pdb(pdb_text):
    """
    Compute per-residue RSA from a PDB string using BioPython ShrakeRupley.
    Returns list of dicts: [{resnum, resname, aa, sasa, rsa}, ...].
    """
    from Bio.PDB import PDBParser, ShrakeRupley

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_text))
    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    results = []
    for chain in structure[0]:
        for residue in chain:
            resname = residue.get_resname()
            if resname not in MAX_ASA:
                continue
            hetflag, resnum, _ = residue.get_id()
            if hetflag.strip():
                continue
            sasa = residue.sasa
            rsa = sasa / MAX_ASA[resname] if MAX_ASA[resname] > 0 else 0.0
            results.append({
                "resnum": resnum,
                "resname": resname,
                "aa": THREE_TO_ONE.get(resname, "X"),
                "sasa": round(sasa, 2),
                "rsa": round(min(rsa, 1.5), 4),
            })
    return results


def map_rsa_to_alignment(aligned_seq, rsa_data):
    """
    Map per-residue RSA values onto alignment columns.
    Returns dict mapping alignment column index (0-based) → RSA value.
    """
    col_to_rsa = {}
    residue_idx = 0
    for col, char in enumerate(aligned_seq):
        if char in "-._":
            continue
        if residue_idx < len(rsa_data):
            col_to_rsa[col] = rsa_data[residue_idx]["rsa"]
            residue_idx += 1
        else:
            break
    return col_to_rsa


def extract_ca_coords(pdb_text):
    """
    Extract Cα coordinates from a PDB string.
    Returns list of dicts: [{resnum, chain, aa, x, y, z}, ...].
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", StringIO(pdb_text))
    residues = []
    for chain in structure[0]:
        for residue in chain:
            hetflag, resnum, _ = residue.get_id()
            if hetflag.strip() or "CA" not in residue:
                continue
            x, y, z = residue["CA"].get_vector()
            residues.append({
                "resnum": resnum,
                "chain": chain.id,
                "aa": THREE_TO_ONE.get(residue.get_resname(), "X"),
                "x": x, "y": y, "z": z,
            })
    return residues


def compute_distance_matrix(residues):
    """Compute L×L pairwise Cα distance matrix (Angstroms) via vectorized numpy."""
    coords = np.array([[r["x"], r["y"], r["z"]] for r in residues], dtype=np.float32)
    sq = (coords ** 2).sum(axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2 * (coords @ coords.T)
    np.clip(dist2, 0, None, out=dist2)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 0.0)
    return dist


def run_rsa_analysis(hits_csv, alignment_fasta, output_dir, cache_dir=None,
                     max_structures=50, include_query=False, conservation_csv=None,
                     prefix=None, redo=False):
    """
    Compute mean RSA per reference position from AlphaFold structures.

    Downloads structures for sequences in the MSA, computes RSA via ShrakeRupley,
    maps onto alignment columns, and writes a per-position summary CSV.
    Skips computation if output exists and redo=False.

    Returns path to the output CSV.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or out_dir.name
    output_path = out_dir / f"{prefix}_rsa_vs_conservation.csv"

    if output_path.exists() and not redo:
        print(f"  [{out_dir.name}] Loading cached RSA")
        return output_path

    if cache_dir is None:
        cache_dir = str(out_dir / "afdb_cache")

    print("Loading BLAST hits and alignment...")
    hits = []
    with open(hits_csv) as f:
        hits = list(csv.DictReader(f))
    aln_records = parse_fasta(alignment_fasta)
    aln_len = len(aln_records[0][2])
    print(f"  {len(hits)} hits, {len(aln_records)} sequences, alignment length {aln_len}")

    col_rsa_values = {col: [] for col in range(aln_len)}
    n_success = 0
    n_fail = 0
    start_idx = 0 if include_query else 1

    print(f"Downloading AlphaFold structures (max {max_structures})...")
    for sid, _, aligned_seq in aln_records[start_idx:]:
        if n_success >= max_structures:
            break
        uniprot_acc = extract_uniprot_accession(sid)
        pdb_text = download_alphafold_structure(uniprot_acc, cache_dir=cache_dir)
        if pdb_text is None:
            n_fail += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rsa_data = compute_rsa_from_pdb(pdb_text)
        except Exception as e:
            print(f"  ERROR computing RSA for {uniprot_acc}: {e}", file=sys.stderr)
            n_fail += 1
            continue
        for col, rsa_val in map_rsa_to_alignment(aligned_seq, rsa_data).items():
            col_rsa_values[col].append(rsa_val)
        n_success += 1
        if n_success % 10 == 0 or n_success <= 3:
            print(f"  Processed {n_success} structures ({n_fail} failed)...")
        time.sleep(0.1)

    print(f"  Total: {n_success} processed, {n_fail} failed")

    ref_seq = aln_records[0][2]
    ref_pos = 0
    results = []
    for col in range(aln_len):
        if ref_seq[col] in "-._":
            continue
        ref_pos += 1
        rsa_vals = col_rsa_values[col]
        if rsa_vals:
            row = {
                "ref_position": ref_pos,
                "consensus": ref_seq[col].upper(),
                "mean_rsa": round(float(np.mean(rsa_vals)), 4),
                "median_rsa": round(float(np.median(rsa_vals)), 4),
                "std_rsa": round(float(np.std(rsa_vals)), 4),
                "n_structures": len(rsa_vals),
            }
        else:
            row = {"ref_position": ref_pos, "consensus": ref_seq[col].upper(),
                   "mean_rsa": "", "median_rsa": "", "std_rsa": "", "n_structures": 0}
        results.append(row)

    if conservation_csv and Path(conservation_csv).exists():
        cons_data = {}
        with open(conservation_csv) as f:
            for row in csv.DictReader(f):
                cons_data[int(row["ref_position"])] = row
        for r in results:
            cd = cons_data.get(r["ref_position"], {})
            r["identity"] = cd.get("identity", "")
            r["entropy"] = cd.get("entropy", "")

    fieldnames = ["ref_position", "consensus", "mean_rsa", "median_rsa", "std_rsa", "n_structures"]
    if conservation_csv:
        fieldnames += ["identity", "entropy"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {output_path}")
    return output_path


def run_ca_distances(accession, output_dir, cache_dir=None, prefix=None,
                     save_csv=True, redo=False):
    """
    Download AlphaFold structure and compute the Cα distance matrix.
    Skips computation if output exists and redo=False.
    Returns path to the .npy output.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or accession
    npy_path = out_dir / f"{prefix}_ca_distances.npy"

    if npy_path.exists() and not redo:
        print(f"  [{accession}] Loading cached Cα distances")
        return npy_path

    if cache_dir is None:
        cache_dir = str(out_dir / "afdb_cache")

    print(f"Downloading AlphaFold structure for {accession}...")
    pdb_text = download_alphafold_structure(accession, cache_dir=cache_dir)
    if pdb_text is None:
        raise RuntimeError(f"Could not download AlphaFold structure for {accession}")

    print("Extracting Cα coordinates...")
    residues = extract_ca_coords(pdb_text)
    print(f"  {len(residues)} residues")

    print("Computing distance matrix...")
    dist = compute_distance_matrix(residues)
    print(f"  Shape: {dist.shape}, range: {dist[dist > 0].min():.2f}–{dist.max():.2f} Å")

    np.save(npy_path, dist)

    res_path = out_dir / f"{prefix}_ca_residues.csv"
    with open(res_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["resnum", "chain", "aa"])
        w.writeheader()
        for r in residues:
            w.writerow({k: r[k] for k in ["resnum", "chain", "aa"]})

    if save_csv:
        resnums = [r["resnum"] for r in residues]
        csv_path = out_dir / f"{prefix}_ca_distances.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["resnum"] + resnums)
            for i, rn in enumerate(resnums):
                w.writerow([rn] + [round(float(v), 3) for v in dist[i]])

    return npy_path
