#!/usr/bin/env python3
"""
Pipeline: Protein conservation from a single query sequence.

    1. Fetch query sequence from UniProt
    2. BLAST against SwissProt (via NCBI)
    3. Filter hits by PID threshold
    4. Fetch hit sequences
    5. Align with MAFFT
    6. Compute per-position conservation

Usage:
    python protein_conservation_pipeline.py --uniprot P00004 --min_pid 50 --max_seqs 200

Requirements:
    pip install biopython requests
    mafft must be installed (conda install -c bioconda mafft)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesession.protein.conservation import run_protein_conservation


def main():
    parser = argparse.ArgumentParser(
        description="Protein conservation pipeline: UniProt → BLAST → MAFFT → conservation"
    )
    parser.add_argument("--uniprot", type=str, default="P00004",
                        help="UniProt accession (default: P00004)")
    parser.add_argument("--min_pid", type=float, default=50.0,
                        help="Minimum percent identity for BLAST hits (default: 50)")
    parser.add_argument("--max_seqs", type=int, default=200,
                        help="Maximum number of sequences to keep (default: 200)")
    parser.add_argument("--database", type=str, default="swissprot",
                        help="BLAST database (default: swissprot)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {cwd}/{uniprot}/)")
    parser.add_argument("--alignment", type=str, default=None,
                        help="Skip steps 1-5, use this pre-existing alignment")
    parser.add_argument("--covariance", action="store_true",
                        help="Compute L×L position-position covariance matrix")
    parser.add_argument("--redo", action="store_true",
                        help="Recompute even if cached outputs exist")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / args.uniprot
    run_protein_conservation(
        accession=args.uniprot,
        output_dir=out_dir,
        min_pid=args.min_pid,
        max_seqs=args.max_seqs,
        database=args.database,
        alignment=args.alignment,
        covariance=args.covariance,
        redo=args.redo,
    )


if __name__ == "__main__":
    main()
