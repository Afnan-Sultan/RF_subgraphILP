import logging
from functools import cached_property

import pandas as pd
from manager.data.gene_matrix_processing import mapped_gene_matrix, render_mat
from manager.data.network_processing import fetch_target_nodes, mapped_nodes

logger = logging.getLogger(__name__)


class ProcessedFiles:
    def __init__(self, files):
        self.files = files
        self.discretized_matrix = pd.read_csv(
            files.discretized_gdsc_file, sep="\t", index_col=0
        )
        self.ic50_matrix = pd.read_csv(files.ic50_gdsc_file, sep="\t", index_col=0)
        self.thresholds = pd.read_csv(files.thresholds_file, sep="\t", index_col=0)
        self.ctd_mapping = pd.read_csv(files.ctd_file)
        self.entrez_symbols = self.ctd_mapping[
            ["GeneID", "GeneSymbol"]
        ].drop_duplicates()
        self.node_entrez, self.aggregates = mapped_nodes(
            files.node_entrez_file, files.aggregate_file
        )
        self.kegg_hsa = pd.read_csv(
            files.kegg_hsa_file, sep="\t", names=["source", "edge", "sink"]
        )
        self.temp_gene_matrix, self.gene_matrix = self.reduce_matrix()

    @cached_property
    def drugs_targets(self):
        targets_file_lines = open(self.files.drugs_targets_file).readlines()[1:]
        return fetch_target_nodes(
            targets_file_lines, self.node_entrez, self.entrez_symbols, self.kegg_hsa
        )

    def reduce_matrix(self):
        gene_matrix = render_mat(self.files.gene_matrix_file)
        return mapped_gene_matrix(gene_matrix, self.entrez_symbols, self.node_entrez)
