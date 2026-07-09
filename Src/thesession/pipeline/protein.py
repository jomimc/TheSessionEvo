"""Protein comparison data: BLAST -> MAFFT -> conservation -> RSA -> Cα distances."""

from thesession.config import PATH_PROTEIN
from thesession.protein import conservation as prot_conservation
from thesession.protein import structure as prot_structure


###################################################################################################
### Protein comparison data

def data_for_protein_comparison(accessions=None, redo=False):
    """
    Compute conservation, RSA, and Cα distances for comparison proteins.

    Runs the full protein pipeline (BLAST → MAFFT → conservation → RSA → Cα distances)
    for each accession.  Cached outputs are reused unless redo=True.

    Parameters
    ----------
    accessions : list of str, optional
        UniProt accessions to process.  Defaults to the three proteins used in the paper.
    redo : bool, optional
        If True, recompute even if cached outputs exist.  Default is False.
    """
    if accessions is None:
        accessions = ['P00004', 'P0AE67', 'P0AA25']

    output_base = PATH_PROTEIN
    cache_dir = str(output_base / "afdb_cache")

    for acc in accessions:
        out_dir = output_base / acc
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Protein: {acc} ---")

        results = prot_conservation.run_protein_conservation(acc, out_dir, redo=redo)
        prot_structure.run_rsa_analysis(
            hits_csv=out_dir / f"{acc}_blast_hits.csv",
            alignment_fasta=results["alignment"],
            output_dir=out_dir,
            cache_dir=cache_dir,
            conservation_csv=results["conservation_csv"],
            redo=redo,
        )
        prot_structure.run_ca_distances(acc, out_dir, cache_dir=cache_dir, redo=redo)
