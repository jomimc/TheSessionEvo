"""MMseqs2 alignment-parameter grid search across the three tune collections."""

from thesession.config import PATH_MMSEQS
from thesession.io import tune_loader as load_tunes
from thesession.io import savage_loader as savage
from thesession.analysis import substitution as SM
from thesession.analysis import optimization as OPT


###################################################################################################
### PARAMETER OPTIMISATION


def run_parameter_optimization(redo=False):
    """
    Generate substitution matrices (if needed) and run MMseqs2 grid-search
    parameter optimisation for all three datasets.

    Calls ``SM.generate_all_sub_mat`` to write substitution matrix files,
    then calls ``optimization.explore_parameter_space`` for TheSession tunes,
    Meertens, and Savage et al. (English), using the same DataFrames that
    are used for Figure 1.

    Parameters
    ----------
    redo : bool, optional
        Passed to the data-loading functions.  Default is ``False``.
    """
    # Generate substitution matrices if they don't exist yet (or on redo)
    path_submat = PATH_MMSEQS.joinpath("substitution_matrices")
    if redo or not any(path_submat.glob("*.out")):
        print("Generating substitution matrices...")
        SM.generate_all_sub_mat()

    df = load_tunes.load_thesession_tunes(redo=redo)
    OPT.explore_parameter_space(df, 'thesession_tunes')

    df = load_tunes.load_meertens_data(redo=redo)[0]
    OPT.explore_parameter_space(df, 'meertens')

    df = savage.load_savage_df(full=True, redo=redo)
    df = df.loc[df.Language == 'English']
    OPT.explore_parameter_space(df, 'savage_english')
