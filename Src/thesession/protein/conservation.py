"""Protein sequence conservation: BLAST, MAFFT alignment, per-position metrics."""

import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .utils import parse_fasta


def fetch_uniprot_fasta(accession):
    """Fetch a single protein sequence from UniProt in FASTA format."""
    import requests
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def run_blast(query_seq, database="swissprot", evalue=1e-5, max_hits=500):
    """Run BLAST against NCBI remotely. Returns parsed BLAST records."""
    from Bio.Blast import NCBIWWW, NCBIXML
    print(f"  Running BLAST against {database} (this may take 1-3 minutes)...")
    result_handle = NCBIWWW.qblast(
        "blastp", database, query_seq,
        expect=evalue, hitlist_size=max_hits, format_type="XML",
    )
    print("  BLAST complete. Parsing results...")
    return list(NCBIXML.parse(result_handle))


def extract_hits(blast_records, min_pid=50.0, min_coverage=0.8, query_length=None):
    """Extract and filter BLAST hit accessions by PID and query coverage."""
    hits = []
    seen = set()
    for record in blast_records:
        if query_length is None:
            query_length = record.query_length
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                pid = (hsp.identities / hsp.align_length) * 100
                coverage = hsp.align_length / query_length
                if pid >= min_pid and coverage >= min_coverage:
                    acc = alignment.accession
                    if acc not in seen:
                        seen.add(acc)
                        hits.append({
                            "accession": acc,
                            "title": alignment.title,
                            "pid": round(pid, 1),
                            "coverage": round(coverage, 3),
                            "evalue": hsp.expect,
                        })
    hits.sort(key=lambda x: -x["pid"])
    return hits


def fetch_sequences_ncbi(accessions, batch_size=50):
    """Fetch protein sequences from NCBI Entrez in batches."""
    from Bio import Entrez
    Entrez.email = "user@example.com"
    all_records = []
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        print(f"  Fetching sequences {i+1}-{min(i+batch_size, len(accessions))} "
              f"of {len(accessions)}...")
        handle = Entrez.efetch(db="protein", id=",".join(batch),
                               rettype="fasta", retmode="text")
        text = handle.read()
        handle.close()
        all_records.extend(parse_fasta(text))
        if i + batch_size < len(accessions):
            time.sleep(0.5)
    return all_records


def align_with_mafft(fasta_path, output_path):
    """Run MAFFT on a FASTA file and write the alignment."""
    print(f"  Aligning {fasta_path} with MAFFT...")
    result = subprocess.run(
        ["mafft", "--auto", "--quiet", str(fasta_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"MAFFT error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    Path(output_path).write_text(result.stdout)
    n = result.stdout.count(">")
    print(f"  Alignment saved: {output_path} ({n} sequences)")
    return Path(output_path)


def compute_conservation(alignment_path, gap_chars=set("-._")):
    """
    Compute per-column conservation from an aligned FASTA file.
    Returns (list of dicts, n_sequences). Reference positions are
    numbered from the first (query) sequence.
    """
    records = parse_fasta(alignment_path)
    if not records:
        raise ValueError("Empty alignment")
    seqs = [seq for _, _, seq in records]
    n_seqs = len(seqs)
    results = []
    ref_pos = 0
    for col in range(len(seqs[0])):
        residues = [s[col].upper() for s in seqs]
        n_gaps = sum(1 for r in residues if r in gap_chars)
        non_gap = [r for r in residues if r not in gap_chars]
        if residues[0] not in gap_chars:
            ref_pos += 1
            rp = ref_pos
        else:
            rp = None
        if not non_gap:
            results.append(dict(ref_position=rp, alignment_col=col+1,
                                entropy=0.0, identity=0.0,
                                gap_fraction=1.0, consensus="-"))
            continue
        counts = Counter(non_gap)
        n_total = len(non_gap)
        freqs = {aa: c / n_total for aa, c in counts.items()}
        entropy = -sum(f * math.log2(f) for f in freqs.values() if f > 0)
        best_aa, best_count = counts.most_common(1)[0]
        results.append(dict(
            ref_position=rp, alignment_col=col+1,
            entropy=round(entropy, 4),
            identity=round(best_count / n_total, 4),
            gap_fraction=round(n_gaps / n_seqs, 4),
            consensus=best_aa,
        ))
    return results, n_seqs


def compute_covariance_matrix(alignment_path, gap_threshold=0.5, gap_chars="-._"):
    """
    Compute an L×L position-position covariance matrix from an MSA.
    Methodology mirrors the melody covariance analysis in the paper.
    Returns (cov, ref_positions, n_pairs).
    """
    records = parse_fasta(alignment_path)
    if not records:
        raise ValueError("Empty alignment")
    seqs = [seq.upper() for _, _, seq in records]
    n_seqs = len(seqs)
    ref_seq = seqs[0]
    gap_set = set(gap_chars)
    ref_cols = [col for col in range(len(ref_seq)) if ref_seq[col] not in gap_set]
    if not ref_cols:
        raise ValueError("Reference sequence has no non-gap positions")
    res_mat = np.array([[seqs[i][col] for col in ref_cols] for i in range(n_seqs)])
    is_gap = np.zeros(res_mat.shape, dtype=bool)
    for g in gap_set:
        is_gap |= (res_mat == g)
    valid = is_gap.mean(axis=0) <= gap_threshold
    res_mat = res_mat[:, valid]
    is_gap = is_gap[:, valid]
    ref_positions = [pos + 1 for pos, keep in enumerate(valid) if keep]
    L_valid = res_mat.shape[1]
    n_pairs = n_seqs * (n_seqs - 1) // 2
    B = np.zeros((n_pairs, L_valid), dtype=np.float32)
    pair_idx = 0
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            both = ~is_gap[i] & ~is_gap[j]
            B[pair_idx] = (res_mat[i] == res_mat[j]) & both
            pair_idx += 1
    cov = np.cov(B.T)
    return cov, ref_positions, n_pairs


def run_protein_conservation(accession, output_dir, min_pid=50.0, max_seqs=200,
                              database="swissprot", alignment=None,
                              covariance=False, redo=False):
    """
    Run the full protein conservation pipeline for a single accession.

    Steps: UniProt fetch → BLAST → filter → NCBI fetch → MAFFT → conservation.
    Pass alignment= to skip steps 1-5 and use a pre-existing aligned FASTA.
    Caches to output_dir; skips computation if outputs exist and redo=False.

    Returns dict with paths: conservation_csv, alignment, and optionally covariance_npy.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = accession

    conservation_csv = out / f"{prefix}_conservation.csv"
    aln_path = out / f"{prefix}_aligned.fasta"

    if conservation_csv.exists() and not redo:
        print(f"  [{accession}] Loading cached conservation")
        return {"conservation_csv": conservation_csv, "alignment": aln_path}

    if alignment:
        aln_path = Path(alignment)
        print(f"  [{accession}] Using pre-existing alignment: {aln_path}")
    else:
        print(f"\n[{accession}] Step 1: Fetching from UniProt")
        query_fasta = fetch_uniprot_fasta(accession)
        query_id, query_desc, query_seq = parse_fasta(query_fasta)[0]
        print(f"  {query_id}: {len(query_seq)} aa")

        print(f"[{accession}] Step 2: BLAST against {database}")
        blast_records = run_blast(query_seq, database=database)

        print(f"[{accession}] Step 3: Filtering hits (PID >= {min_pid}%)")
        hits = extract_hits(blast_records, min_pid=min_pid, query_length=len(query_seq))
        if not hits:
            raise ValueError(f"No BLAST hits found for {accession} at PID >= {min_pid}%")
        hits = hits[:max_seqs]
        print(f"  {len(hits)} hits kept")

        hits_path = out / f"{prefix}_blast_hits.csv"
        with open(hits_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["accession", "pid", "coverage", "evalue", "title"])
            w.writeheader()
            w.writerows(hits)

        print(f"[{accession}] Step 4: Fetching sequences from NCBI")
        fetched = fetch_sequences_ncbi([h["accession"] for h in hits])
        print(f"  Fetched {len(fetched)} sequences")

        unaligned_path = out / f"{prefix}_unaligned.fasta"
        with open(unaligned_path, "w") as f:
            f.write(f">{query_id} {query_desc}\n")
            for i in range(0, len(query_seq), 80):
                f.write(query_seq[i:i+80] + "\n")
            for sid, desc, seq in fetched:
                f.write(f">{sid} {desc}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")

        print(f"[{accession}] Step 5: Aligning with MAFFT")
        aln_path = align_with_mafft(str(unaligned_path), str(out / f"{prefix}_aligned.fasta"))

    print(f"[{accession}] Step 6: Computing conservation")
    conservation, n_seqs = compute_conservation(aln_path)
    ref_data = [c for c in conservation if c["ref_position"] is not None]
    print(f"  {len(ref_data)} reference positions, {n_seqs} sequences")

    with open(conservation_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ref_position", "consensus", "identity",
                                           "entropy", "gap_fraction"])
        w.writeheader()
        for row in ref_data:
            w.writerow({k: row[k] for k in w.fieldnames})

    json_path = out / f"{prefix}_conservation.json"
    with open(json_path, "w") as f:
        json.dump({"query": accession, "n_sequences": n_seqs,
                   "n_positions": len(ref_data), "per_position": ref_data}, f, indent=2)

    results = {"conservation_csv": conservation_csv, "alignment": aln_path}

    if covariance:
        print(f"[{accession}] Step 7: Computing covariance matrix")
        cov, ref_positions, n_pairs = compute_covariance_matrix(aln_path)
        npy_path = out / f"{prefix}_covariance.npy"
        np.save(npy_path, cov)
        results["covariance_npy"] = npy_path
        print(f"  {len(ref_positions)}×{len(ref_positions)} matrix ({n_pairs} pairs)")

    return results
