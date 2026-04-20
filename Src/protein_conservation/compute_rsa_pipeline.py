#!/usr/bin/env python3
"""
Compute per-alignment-column RSA from AlphaFold DB structures.

Takes as input the outputs of protein_conservation_pipeline.py:
  - *_blast_hits.csv  (accession list)
  - *_aligned.fasta   (MSA)

Downloads AlphaFold structures, computes RSA, maps onto alignment columns,
and outputs mean RSA per reference position alongside identity.

Usage:
    python compute_rsa_pipeline.py \
        --hits P00004_blast_hits.csv \
        --alignment P00004_aligned.fasta \
        --conservation P00004_conservation.csv \
        --output_dir P00004/

Requirements:
    pip install biopython requests
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesession.protein.structure import run_rsa_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-site RSA from AlphaFold structures, mapped to MSA"
    )
    parser.add_argument("--hits", required=True,
                        help="BLAST hits CSV from protein_conservation_pipeline.py")
    parser.add_argument("--alignment", required=True,
                        help="Aligned FASTA from protein_conservation_pipeline.py")
    parser.add_argument("--conservation", default=None,
                        help="Conservation CSV for merging (optional)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: inferred from --hits filename)")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to cache downloaded PDB files")
    parser.add_argument("--max_structures", type=int, default=50,
                        help="Max number of structures to download (default: 50)")
    parser.add_argument("--include_query", action="store_true",
                        help="Also compute RSA for the query sequence")
    parser.add_argument("--redo", action="store_true",
                        help="Recompute even if cached output exists")
    args = parser.parse_args()

    hits_stem = Path(args.hits).name
    inferred_acc = hits_stem.split("_")[0] if "_" in hits_stem else hits_stem
    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / inferred_acc

    run_rsa_analysis(
        hits_csv=args.hits,
        alignment_fasta=args.alignment,
        output_dir=out_dir,
        cache_dir=args.cache_dir,
        max_structures=args.max_structures,
        include_query=args.include_query,
        conservation_csv=args.conservation,
        redo=args.redo,
    )


if __name__ == "__main__":
    main()
