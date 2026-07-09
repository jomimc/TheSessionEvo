"""Download + unpack Zenodo data tiers into the repo root.

Usage:
    python scripts/download_data.py --tier figures   # tier 1 only (default)
    python scripts/download_data.py --tier full       # tiers 1 + 2

Works on Windows/macOS/Linux — stdlib only (urllib + zipfile), no bash.
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Filled in on release with the Zenodo record's per-file download URLs, e.g.:
#   TIER1_URL = "https://zenodo.org/records/<id>/files/tier1_figuredata.zip"
#   TIER2_URL = "https://zenodo.org/records/<id>/files/tier2_caches.zip"
TIER1_URL = None
TIER2_URL = None
MANIFEST_URL = None  # "https://zenodo.org/records/<id>/files/MANIFEST.sha256"

TIERS = {
    "figures": [("tier1_figuredata.zip", lambda: TIER1_URL)],
    "full": [
        ("tier1_figuredata.zip", lambda: TIER1_URL),
        ("tier2_caches.zip", lambda: TIER2_URL),
    ],
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest():
    if not MANIFEST_URL:
        return {}
    tmp = REPO_ROOT / "MANIFEST.sha256"
    urllib.request.urlretrieve(MANIFEST_URL, tmp)
    checksums = {}
    for line in tmp.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            digest, fname = parts
            checksums[fname.lstrip("*")] = digest
    return checksums


def download_and_unpack(fname, url):
    if url is None:
        print(f"ERROR: no download URL configured for {fname} yet.", file=sys.stderr)
        print("  This repo's scripts/download_data.py needs its TIER*_URL", file=sys.stderr)
        print("  constants filled in with the Zenodo record's file URLs.", file=sys.stderr)
        sys.exit(1)

    dest = REPO_ROOT / fname
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

    manifest = _load_manifest()
    if fname in manifest:
        digest = _sha256(dest)
        if digest != manifest[fname]:
            print(f"ERROR: checksum mismatch for {fname}", file=sys.stderr)
            print(f"  expected {manifest[fname]}", file=sys.stderr)
            print(f"  got      {digest}", file=sys.stderr)
            sys.exit(1)
        print(f"  Checksum OK ({digest[:16]}...)")

    print(f"Unpacking {fname} into {REPO_ROOT}")
    with zipfile.ZipFile(dest) as zf:
        zf.extractall(REPO_ROOT)
    dest.unlink()
    print(f"Done: {fname}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=sorted(TIERS), default="figures",
                         help="Which tier(s) to download (default: figures).")
    args = parser.parse_args()

    for fname, url_getter in TIERS[args.tier]:
        download_and_unpack(fname, url_getter())

    print("\nAll requested tiers unpacked. See README_DATA.md for what each tier contains.")


if __name__ == "__main__":
    main()
