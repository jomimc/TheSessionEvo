"""Shared utilities for protein conservation analysis."""

import sys
from pathlib import Path


MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_fasta(source):
    """
    Parse FASTA from a file path, Path object, or raw FASTA string.
    Returns list of (id, description, sequence) tuples.
    """
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        text = Path(source).read_text()
    else:
        text = str(source)
    records = []
    current_id = None
    current_desc = ""
    current_seq = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, current_desc, "".join(current_seq)))
            parts = line[1:].split(None, 1)
            current_id = parts[0]
            current_desc = parts[1] if len(parts) > 1 else ""
            current_seq = []
        elif line:
            current_seq.append(line)
    if current_id is not None:
        records.append((current_id, current_desc, "".join(current_seq)))
    return records


def download_alphafold_structure(uniprot_acc, cache_dir=None, verbose=True):
    """
    Download AlphaFold structure PDB text for a UniProt accession.
    Checks cache first; saves to cache after download.
    Returns PDB text string, or None if unavailable.
    """
    import requests

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for cached in Path(cache_dir).glob(f"AF-{uniprot_acc}-*.pdb"):
            if verbose:
                print(f"  [cache] {uniprot_acc}")
            return cached.read_text()

    headers = {"User-Agent": "protein-conservation-pipeline/1.0"}
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_acc}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"  [FAIL] {uniprot_acc}: API request error: {e}", file=sys.stderr)
        return None

    if resp.status_code != 200:
        if verbose:
            print(f"  [FAIL] {uniprot_acc}: API returned HTTP {resp.status_code}", file=sys.stderr)
        return None

    try:
        data = resp.json()
    except Exception as e:
        if verbose:
            print(f"  [FAIL] {uniprot_acc}: JSON parse error: {e}", file=sys.stderr)
        return None

    if not data:
        if verbose:
            print(f"  [FAIL] {uniprot_acc}: empty API response", file=sys.stderr)
        return None

    prediction = data[0] if isinstance(data, list) else data
    pdb_url = prediction.get("pdbUrl") or prediction.get("pdb_url")

    if not pdb_url:
        entry_id = prediction.get("entryId", f"AF-{uniprot_acc}-F1")
        raw_version = prediction.get("latestVersion", 4)
        model_version = f"v{raw_version}" if isinstance(raw_version, int) else raw_version
        pdb_url = f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_{model_version}.pdb"
        if verbose:
            print(f"  [warn] {uniprot_acc}: no pdbUrl in API, using fallback: {pdb_url}",
                  file=sys.stderr)

    try:
        pdb_resp = requests.get(pdb_url, headers=headers, timeout=60)
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"  [FAIL] {uniprot_acc}: PDB download error: {e}", file=sys.stderr)
        return None

    if pdb_resp.status_code != 200:
        if verbose:
            print(f"  [FAIL] {uniprot_acc}: PDB HTTP {pdb_resp.status_code} for {pdb_url}",
                  file=sys.stderr)
        return None

    pdb_text = pdb_resp.text

    if cache_dir:
        entry_id = prediction.get("entryId", f"AF-{uniprot_acc}-F1")
        (Path(cache_dir) / f"{entry_id}.pdb").write_text(pdb_text)

    return pdb_text
