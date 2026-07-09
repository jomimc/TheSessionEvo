# Obtaining the Meertens Tune Collections

The Meertens Tune Collections (Dutch folk music) are **form-gated** — they cannot be
redistributed or auto-fetched, and `scripts/fetch_data.py` does not handle this source.

## Manual download

1. Request access at [liederenbank.nl/mtc](https://www.liederenbank.nl/mtc).
2. Once granted, download the **MTC-FS-INST-2.0** subset.
3. Unpack it so the following paths exist under `Data/`:

   ```
   Data/Meertens/MTC-FS-INST-2.0/krn/                                    (kern files)
   Data/Meertens/MTC-FS-INST-2.0/metadata/MTC-FS-INST-2.0.csv
   Data/Meertens/MTC-FS-INST-2.0/metadata/MTC-FS-INST-2.0-fieldnames.csv
   ```

   (`Data/Meertens` may be a real directory or a symlink to wherever you keep the
   corpus — `THESESSION_DATA` also works if you prefer to keep raw data out-of-tree.)

## Skipping this step

You don't need the raw Meertens corpus to reproduce any published figure — the parsed
`meertens_tunes.pkl` / `meertens_summary.pkl` caches are shipped in the Zenodo **Tier 2**
archive (see `README_DATA.md`). The raw corpus is only needed if you want to re-parse
Meertens from scratch (e.g. after a metadata correction upstream).
