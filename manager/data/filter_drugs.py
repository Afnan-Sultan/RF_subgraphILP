import logging
import os

import pandas as pd
from manager.config import Kwargs
from tqdm import tqdm

logger = logging.getLogger(__name__)


def select_by_drug(discretized_mat, num_cell_lines):
    """
    identify drugs that have been experimented on > thresh cell lines and return these cell lines
    """
    drug_cell_dict = {}  # store cell lines associated with a drug
    for drug in discretized_mat:
        # ignore cell lines with NaN
        used_cell_lines = discretized_mat[drug].dropna().index.to_list()
        num_cells = len(used_cell_lines)
        if num_cells >= num_cell_lines:
            drug_cell_dict[drug] = used_cell_lines
    return drug_cell_dict


def filter_drugs(kwargs: Kwargs):
    """
    Select gene expression matrix corresponding to the cell lines associated with drugs passing the cell lines threshold
    """
    selected_drugs = {}
    if kwargs.from_disk and not kwargs.training.original_mat:
        # TODO: can be removed when done. used only for repetition sake
        matrices_dir = kwargs.matrices_output_dir
        if kwargs.training.target_root_node:
            to_upload = [
                drug
                for drug in os.listdir(matrices_dir)
                if drug in kwargs.data.processed_files.drugs_targets.keys()
            ]
        else:
            to_upload = os.listdir(matrices_dir)

        drug_subset = kwargs.data.drug_subset
        if drug_subset is not None:
            if isinstance(drug_subset, int):
                to_upload = to_upload[:drug_subset]
            else:
                if kwargs.data.include_subset:
                    to_upload = [drug for drug in to_upload if drug in drug_subset]
                else:
                    to_upload = [drug for drug in to_upload if drug not in drug_subset]

        for drug_name in tqdm(
            to_upload,
        ):
            selected_drugs[drug_name] = {
                "gene_mat": pd.read_csv(
                    os.path.join(matrices_dir, drug_name, "gene_mat.txt"),
                    sep="\t",
                    index_col=0,
                ),
                "meta_data": pd.read_csv(
                    os.path.join(matrices_dir, drug_name, "meta.txt"),
                    sep="\t",
                    index_col=0,
                ),
            }
    else:
        # fetch the drugs passing the cell lines threshold and these cell lines
        drugs_cells = select_by_drug(
            kwargs.data.processed_files.discretized_matrix,
            kwargs.training.cell_lines_thresh,
        )

        # fetch the gene expression matrix for the retrieved cell lines for each drug
        for drug, cell_lines in tqdm(
            drugs_cells.items(), desc="selecting individual gene matrix per drug ..."
        ):
            # not all samples reported from discretized_matrix are available in the gene matrix! TODO: expected?!
            present_cell_lines = list(
                set(kwargs.data.processed_files.gene_matrix.columns).intersection(
                    str(cl) for cl in cell_lines
                )
            )
            if kwargs.training.original_mat:
                drug_df = kwargs.data.processed_files.original_gene_matrix.loc[
                    :, present_cell_lines
                ]
            else:
                drug_df = kwargs.data.processed_files.gene_matrix.loc[
                    :, present_cell_lines
                ]

            present_cell_lines = drug_df.columns.astype(int)
            labels = kwargs.data.processed_files.discretized_matrix.loc[
                present_cell_lines, drug
            ].to_list()
            ic_50 = kwargs.data.processed_files.ic50_matrix.loc[
                present_cell_lines, drug
            ].to_list()
            meta_data = pd.DataFrame(
                {"Labels": labels, "ic_50": ic_50}, index=present_cell_lines
            )

            if kwargs.matrices_output_dir is not None and kwargs.output_selected_drugs:
                drug_output_dir = os.path.join(kwargs.matrices_output_dir, drug)
                os.makedirs(drug_output_dir, exist_ok=True)
                drug_df.to_csv(os.path.join(drug_output_dir, "gene_mat.txt"), sep="\t")
                meta_data.to_csv(os.path.join(drug_output_dir, "meta.txt"), sep="\t")
            selected_drugs[drug] = {"gene_mat": drug_df, "meta_data": meta_data}
    return selected_drugs
