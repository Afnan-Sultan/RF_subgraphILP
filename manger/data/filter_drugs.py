import logging
import os

import pandas as pd
from tqdm import tqdm

from manger.config import Kwargs

logger = logging.getLogger(__name__)


def select_by_drug(discretized_mat, num_cell_lines):
    """
    identify drugs that have been experimented on > thresh cell lines and return these cell lines
    """
    drug_cell_dict = {}  # store cell lines associated with a drug
    for drug in discretized_mat:
        # ignore NaNs from counted cell lines
        used_cell_lines = discretized_mat[
            discretized_mat[drug].notnull()
        ].index.to_list()
        num_cells = len(used_cell_lines)
        if num_cells >= num_cell_lines:
            drug_cell_dict[drug] = used_cell_lines
    return drug_cell_dict


def filter_drugs(kwargs: Kwargs):
    """
    Select the genes scores for each drug, corresponding to the cell lines associated with that drug.
    """
    selected_drugs = {}
    if kwargs.from_disk:
        for drug_name in tqdm(
            os.listdir(kwargs.matrices_output_dir),
            desc="Fetching selected drugs from desk",
        ):
            selected_drugs[drug_name] = {
                "gene_mat": pd.read_csv(
                    os.path.join(kwargs.matrices_output_dir, drug_name, "gene_mat.txt"),
                    sep="\t",
                    index_col=0,
                ),
                "meta_data": pd.read_csv(
                    os.path.join(kwargs.matrices_output_dir, drug_name, "meta.txt"),
                    sep="\t",
                    index_col=0,
                ),
            }
    else:
        drugs_cells = select_by_drug(
            kwargs.data.processed_files.discretized_matrix,
            kwargs.training.cell_lines_thresh,
        )
        for drug, cell_lines in tqdm(
            drugs_cells.items(), desc="selecting individual gene matrix per drug ..."
        ):
            present_cell_lines = sorted(
                list(
                    set(kwargs.data.processed_files.gene_matrix.columns).intersection(
                        [str(cl) for cl in cell_lines]
                    )
                )
            )
            drug_df = kwargs.data.processed_files.gene_matrix.loc[:, present_cell_lines]

            int_index = [int(cl) for cl in present_cell_lines]
            labels = kwargs.data.processed_files.discretized_matrix.loc[
                int_index, drug
            ].to_list()
            ic_50 = kwargs.data.processed_files.ic50_matrix.loc[
                int_index, drug
            ].to_list()
            meta_data = pd.DataFrame(
                {"Labels": labels, "ic_50": ic_50}, index=present_cell_lines
            )

            if kwargs.matrices_output_dir is not None:
                drug_output_dir = os.path.join(kwargs.matrices_output_dir, drug)
                os.makedirs(drug_output_dir, exist_ok=True)
                drug_df.to_csv(os.path.join(drug_output_dir, "gene_mat.txt"), sep="\t")
                meta_data.to_csv(os.path.join(drug_output_dir, "meta.txt"), sep="\t")
            selected_drugs[drug] = {"gene_mat": drug_df, "meta_data": meta_data}
    return selected_drugs
