"""MMseqs2 all-vs-all alignment: run/load results and build the main aligned dataset."""

import shutil
from subprocess import Popen, PIPE
import time

from thesession.config import MMSEQS_BIN, PATH_MMSEQS
from thesession.io import seq_io
from thesession.io import tune_loader as load_tunes
from thesession.alignment import parts as PA
from thesession.structure import part_separation as PS
from thesession.analysis import substitution as SM


### Create folder + files and run mmseqs

### run_mmseqs will not work properly for parts!!!
### To go here or somewhere else? Probably a separate module
def run_mmseqs(df, name, go=4, ge=3, ref='ref', save_fasta=True):
    """
    Write sequences to FASTA and run an MMseqs2 all-vs-all easy-search.

    Creates a subdirectory ``PATH_MMSEQS / name``, writes a FASTA file
    of pitch-class sequences, writes a custom substitution matrix, then
    invokes ``mmseqs easy-search``.  Temporary MMseqs2 files are
    deleted on success.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with at least a ``tchroma`` column (pitch-class sequence)
        and a column named ``ref`` (or the value of ``ref``) for
        sequence identifiers.
    name : str
        Subdirectory name under ``PATH_MMSEQS`` and dataset label used
        for FASTA and result file naming.
    go : int, optional
        Gap-open penalty (positive integer passed to MMseqs2).
        Default is ``4``.
    ge : int, optional
        Gap-extend penalty (positive integer passed to MMseqs2).
        Default is ``3``.
    ref : str, optional
        Column name in ``df`` to use as sequence identifiers.
        Default is ``'ref'``.
    save_fasta : bool, optional
        If ``True`` (default), write the FASTA file before running
        MMseqs2.  Set to ``False`` if the FASTA already exists.

    Returns
    -------
    None

    Notes
    -----
    Raises an exception if ``mmseqs`` is not found on the system PATH.
    MMseqs2 output is in ``--format-mode 4`` (tab-separated with
    headers).  The substitution matrix uses ``SM.basic_submat_A(6, -4)``
    (match score 6, mismatch score -4) for the 12-pitch-class alphabet.
    """
    # First check that mmseqs is installed / in the system's path
    if isinstance(shutil.which('mmseqs'), type(None)):
        raise Exception("MMseqs is not found in the system path. Aborting!")

    # Save to fasta
    path_base = PATH_MMSEQS.joinpath(f'{name}')
    path_base.mkdir(parents=True, exist_ok=True)
    fasta_name = f'all_seq_{name}.fasta'
    if save_fasta:
        path_fasta = path_base.joinpath(fasta_name)
        seq_io.write_all_seq_to_fasta(df.tchroma, df[ref], path_fasta)

    # Save substitution matrix file
    path_submat = PATH_MMSEQS.joinpath(f'{name}/matrix.out')
    submat = SM.basic_submat_A(6, -4)
    SM.write_mmseqs_sub_mat(path_submat, submat, nmax=12)

    # Run mmseqs
    ### Run mmseqs search with inputs:
    ###     fasta (all sequences, for all-vs-all comparison)
    ###     submat (substitution matrix, with match/mismatch scores)
    ###     gap_open, gap_extend (gap penalties, given as positive numbers)
    ### Outputs are saved in path_result, with temporary files in path_tmp
    args = [MMSEQS_BIN, 'easy-search', fasta_name, fasta_name,
            "result.m8", "tmp", '--format-mode', '4',
            '--sub-mat', "matrix.out", '--gap-open', str(go),
            '--gap-extend', str(ge)]

    pipe_output = Popen(args, stdout=PIPE, stderr=PIPE, cwd=str(path_base))
    stdout, stderr = pipe_output.communicate()

    if pipe_output.returncode == 0:
        print("MMseqs has completed successfully!")
        shutil.rmtree(path_base.joinpath("tmp"))
    else:
        print("An error has occurred while running MMseqs!")
        print(stderr.decode())


### Load mmseqs results (or run mmseqs if not done yet)
def load_mmseqs(df, dataset, ref='setting_id', redo=False, annotate=True, save_fasta=True):
    """
    Load MMseqs2 results from cache, running MMseqs2 first if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table passed to ``run_mmseqs`` (if run) and to
        ``seq_io.load_mmseqs_pairwise`` (if ``annotate=True``).
    dataset : str
        Dataset name; determines subdirectory and result file path.
    ref : str, optional
        Sequence-identifier column passed to ``run_mmseqs``.
        Default is ``'setting_id'``.
    redo : bool, optional
        If ``True``, re-run MMseqs2 even if a result file exists.
        Default is ``False``.
    annotate : bool, optional
        Passed to ``seq_io.load_mmseqs_pairwise``; if ``True`` (default),
        adds an ``in_fam`` column.
    save_fasta : bool, optional
        Passed to ``run_mmseqs``; default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Filtered and optionally annotated MMseqs2 result table.
    """
    path = PATH_MMSEQS.joinpath(f"{dataset}/result.m8")
    if not path.exists() or redo:
        print(f"Running mmseqs for '{dataset}'...")
        run_mmseqs(df, dataset, ref=ref, save_fasta=save_fasta)
    else:
        print(f"Loading cached mmseqs results for '{dataset}'")
    return seq_io.load_mmseqs_pairwise(df, dataset, annotate)


###################################################################################################
### Build the main aligned dataset (shared prerequisite for every analysis)

### Runs on: TheSession
### Loads the full cleaned dataset, separates tunes into parts,
### writes parts to fasta, runs mmseqs, and annotates the surviving hit pairs.
def run_main_alignments(redo=False):
    """
    Run the full TheSession part-alignment pipeline.

    This is the central computation step shared by every downstream analysis
    (mutability, substitution, position, covariance).  It:

    1. Loads the cleaned TheSession dataset (``~2 h`` first run).
    2. Splits each setting into structural parts.
    3. Writes parts to FASTA and runs MMseqs2.
    4. Prunes hits from duplicate parts.
    5. Annotates and filters all surviving hit pairs.

    Parameters
    ----------
    redo : bool, optional
        Passed to every sub-step; if ``True``, all caches are
        invalidated and recomputed.  Default is ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned setting-level metadata.
    tunes : dict
        Music21-parsed feature dicts keyed by ``setting_id``.
    df_parts : pandas.DataFrame
        Part-level metadata (one row per extracted part).
    parts_data : dict
        Part feature dicts keyed by ``part_id``.
    res : pandas.DataFrame
        Full annotated MMseqs2 hit table (all surviving pairs).
    res0 : pandas.DataFrame
        Filtered hit table (equal duration, equal meter, 0.5 < fident < 1).
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (substitutions, matches, etc.).
    """
    # Load a cleaned dataset
    # (code takes about 2 hours to run)
    print(f"Loading data (redo is {'on' if redo else 'off'})")
    t0 = time.time()
    df, tunes = load_tunes.load_thesession_data(redo=redo)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Extract parts
    print(f"Splitting tunes into parts")
    t0 = time.time()
    df_parts, parts_data = PS.get_all_parts_thesession(df, tunes, redo=redo)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Write parts to fasta
    print(f"Writing to fasta")
    seq_io.write_parts_thesession(parts_data)

    # (Run and ) load mmseqs2 results
    t0 = time.time()
    res = load_mmseqs(parts_data, "thesession_parts", redo=redo, annotate=False, save_fasta=False)
    print(f"  mmseqs gave {len(res)} tune pairs ({time.time()-t0:.1f}s)")

    res = PA.prune_identical_parts(res, parts_data)
    print(f"Pruning identical parts leaves {len(res)} tune pairs")

    # Align parts using new algorithm
    print("Annotating alignments...")
    t0 = time.time()
    res, res0, mismatches = PA.annotate_res(df, df_parts, res, parts_data, redo=redo)
    print(f"  Final set: {len(res0)} tune pairs ({time.time()-t0:.1f}s)")

    return df, tunes, df_parts, parts_data, res, res0, mismatches
