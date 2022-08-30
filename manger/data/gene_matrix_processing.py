import logging

import pandas as pd

logger = logging.getLogger(__name__)


def render_mat(path):
    """
    a function to render the gene expression matrix, by removing extra and replicated values
    """
    gene_mat = pd.read_csv(path, sep="\t", index_col=0)
    gene_mat.index.name = None  # the file comes with an index name, which is not usable

    # aggregate duplicated gene values by mean
    gene_mat = gene_mat.groupby(gene_mat.index).mean()

    return gene_mat


def mapped_gene_matrix(gene_mat, entrez_symbols, node_entrez):
    """
    Select only scores of the genes that can be mapped to entrez IDs, as these will be the ones to be used
    in subgraphIP. Also, return the matrix with the gene enetrez ids instead of the gene symbols.
    """

    logger.info(f"Number of genes in original matrix = {len(gene_mat)}")

    # Identify gene symbols in the matrix that are in the entrez_symbol dataframe
    index_matching = entrez_symbols[entrez_symbols["GeneSymbol"].isin(gene_mat.index)]
    matched_symb = index_matching["GeneSymbol"].to_list()
    matched_matrix = gene_mat.loc[matched_symb, :]

    # replace the gene symbols with the Entrez IDs
    matched_matrix.index = index_matching["GeneID"].to_list()
    logger.info(f"Number of matched symbols to entrez = {len(matched_matrix)}")

    subgraph_legit_genes = list(
        set(node_entrez["GeneID"].to_list()).intersection(matched_matrix.index)
    )
    final_matrix = matched_matrix.loc[subgraph_legit_genes, :]
    logger.info(f"Number of subgraphILP legit genes = {len(final_matrix)}")
    return matched_matrix, final_matrix  # TODO: remove one of them
