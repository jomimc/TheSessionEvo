#!/usr/bin/env python3
"""
Download an AlphaFold structure and compute the Cα distance matrix.

Usage:
    python ca_distance_matrix.py P00004
    python ca_distance_matrix.py P00004 --output_dir /path/to/dir
    python ca_distance_matrix.py P00004 --no_csv

Outputs (in {cwd}/{uniprot}/ by default):
    {prefix}_ca_distances.npy   — L×L float32 distance matrix (Angstroms)
    {prefix}_ca_distances.csv   — same, with residue-number headers
    {prefix}_ca_residues.csv    — per-residue metadata (resnum, aa, chain)

Requirements:
    pip install biopython numpy requests
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesession.protein.structure import run_ca_distances


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold structure and compute Cα distance matrix"
    )
    parser.add_argument("uniprot", help="UniProt accession (e.g. P00004)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: {cwd}/{uniprot}/)")
    parser.add_argument("--prefix", default=None,
                        help="Output filename prefix (default: uniprot accession)")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache directory for PDB files")
    parser.add_argument("--no_csv", action="store_true",
                        help="Skip CSV output (large for long proteins)")
    parser.add_argument("--redo", action="store_true",
                        help="Recompute even if cached output exists")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / args.uniprot
    run_ca_distances(
        accession=args.uniprot,
        output_dir=out_dir,
        cache_dir=args.cache_dir,
        prefix=args.prefix,
        save_csv=not args.no_csv,
        redo=args.redo,
    )


if __name__ == "__main__":
    main()
