import logging

import pandas as pd
from manager.data.gene_matrix_processing import mapped_gene_matrix, render_mat
from manager.data.network_processing import fetch_target_nodes, mapped_nodes

logger = logging.getLogger(__name__)


class ProcessedFiles:
    def __init__(self, files):
        self.discretized_matrix = self.remove_axis_spaces(
            files.discretized_gdsc_file, 1
        )
        self.ic50_matrix = self.remove_axis_spaces(files.ic50_gdsc_file, 1)
        self.thresholds = self.remove_axis_spaces(files.thresholds_file, 0)
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
        self.original_gene_matrix = render_mat(files.gene_matrix_file)
        self.temp_gene_matrix, self.gene_matrix = mapped_gene_matrix(
            self.original_gene_matrix, self.entrez_symbols, self.node_entrez
        )
        self.gdsc_drugs = self.reformat_gdsc_drugs_file(files.drugs_targets_file)
        self.drugs_targets = fetch_target_nodes(
            self.gdsc_drugs,
            self.node_entrez,
            self.aggregates,
            self.entrez_symbols,
            self.kegg_hsa,
        )

    @staticmethod
    def remove_axis_spaces(file, axis):
        """
        axis is used for naming, and spaces can carry further complications
        """
        df = pd.read_csv(file, sep="\t", index_col=0)
        if axis == 0:
            df.index = df.index.str.replace(" ", "--")
        else:
            df.columns = df.columns.str.replace(" ", "--")
        return df

    @staticmethod
    def reformat_gdsc_drugs_file(drugs_file):
        gdsc_drugs = pd.read_csv(drugs_file).dropna()

        # removing extra spaces
        refined_targets = [
            ",".join([target.strip() for target in targets.split(",")])
            for targets in gdsc_drugs[" Targets"]
        ]
        refined_drugs_names = [name.replace(" ", "--") for name in gdsc_drugs[" Name"]]

        drugs_targets = pd.Series(refined_targets, index=refined_drugs_names)

        return drugs_targets
