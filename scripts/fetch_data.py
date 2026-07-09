"""Fetch the redistributable raw corpora (Tier 3) into Data/, pinned to the
exact commits used in the paper.

Usage:
    python scripts/fetch_data.py            # fetch everything
    python scripts/fetch_data.py --source thesession
    python scripts/fetch_data.py --source savage

Requires only Python 3 and git on PATH (no bash, so this works on Windows).

Two sources are not handled here:
  - Data/SavageFig/*.txt   — digitized from a paper figure, exists in no
                              public repo; shipped in the Zenodo Tier 2 zip.
  - Data/Meertens/         — form-gated (https://www.liederenbank.nl/mtc);
                              see DATA.md for the manual download instructions.
"""

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"

THESESSION_URL = "https://github.com/adactio/TheSession-data.git"
THESESSION_SHA = "4f2d9d551f941caacb9d1f3a37721b319108bce7"

SAVAGE_SHA = "82a8625fae4ecc5ee3141d6e48bf3ae3b554d59b"
SAVAGE_RAW_BASE = f"https://raw.githubusercontent.com/pesavage/melodic-evolution/{SAVAGE_SHA}/data"
SAVAGE_FILES = ["MelodicEvoSeq.xlsx", "MelodicEvoSeqFullSongs.xlsx"]


def fetch_thesession():
    """Clone adactio/TheSession-data and pin it to the frozen commit."""
    dest = DATA_DIR / "TheSession-data"
    if dest.exists():
        print(f"[thesession] {dest} already exists — skipping clone.")
        print(f"  To re-fetch, delete it first: rm -rf {dest}")
        return
    print(f"[thesession] Cloning {THESESSION_URL} -> {dest}")
    subprocess.run(["git", "clone", THESESSION_URL, str(dest)], check=True)
    print(f"[thesession] Checking out pinned commit {THESESSION_SHA}")
    subprocess.run(["git", "checkout", THESESSION_SHA], check=True, cwd=str(dest))
    print("[thesession] Done.")


def fetch_savage():
    """Download the two Savage et al. Excel sheets, pinned to the frozen commit."""
    dest_dir = DATA_DIR / "Bronson"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in SAVAGE_FILES:
        dest = dest_dir / fname
        if dest.exists():
            print(f"[savage] {dest} already exists — skipping.")
            continue
        url = f"{SAVAGE_RAW_BASE}/{fname}"
        print(f"[savage] Downloading {url}")
        urllib.request.urlretrieve(url, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"  Saved {dest} (sha256 {digest[:16]}...)")
    print("[savage] Done.")


SOURCES = {"thesession": fetch_thesession, "savage": fetch_savage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default=None,
                         help="Fetch only this source (default: fetch all).")
    args = parser.parse_args()

    targets = [args.source] if args.source else sorted(SOURCES)
    for name in targets:
        SOURCES[name]()

    print("\nNote: Data/SavageFig/ and Data/Meertens/ are not fetched by this script.")
    print("  SavageFig/*.txt ships in the Zenodo Tier 2 archive (not redistributable elsewhere).")
    print("  Meertens requires a signed form — see DATA.md.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: command failed: {e}", file=sys.stderr)
        sys.exit(1)
