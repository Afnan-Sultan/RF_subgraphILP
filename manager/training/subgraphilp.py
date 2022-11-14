import json
import os.path
import subprocess
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
from manager.config import Kwargs
from manager.data.network_processing import map_nodes_to_entrez
from manager.utils import NewJsonEncoder


def get_samples(gene_mat, classes, labels, cls_as_cols=True):
    """
    select samples belonging to a specified class.
    This function ensures repetition in case of bootstrapped gene matrix
    """

    # get cell lines belonging to each label
    cls_per_label = {}
    for name, label in labels.items():
        cls_per_label[name] = classes[classes.values == label].index.to_list()

    if cls_as_cols:
        to_enumerate = gene_mat.columns.to_list()
    else:
        to_enumerate = gene_mat.index.to_list()

    # ensure repetition of samples in case of bootstrapped gene matrix
    idx_per_label = {}
    for idx, cl in enumerate(to_enumerate):
        for name, cls in cls_per_label.items():
            if cl in cls:
                if name in idx_per_label:
                    idx_per_label[name].append(idx)
                else:
                    idx_per_label[name] = [idx]
                break

    # retrieve the corresponding matrix per label
    mat_per_label = {}
    for name, cls in idx_per_label.items():
        if cls_as_cols:
            label_mat = gene_mat.iloc[:, cls]
        else:
            label_mat = gene_mat.iloc[cls, :]
        mat_per_label[name] = label_mat

    return mat_per_label


def differential_expression(
    gene_mat: pd.DataFrame, classes: pd.Series, output_dir: str, de_method: str
):
    """
    calculate differential expression for a gene matrix
    gene_mat = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    return: str to output file path
    """

    gene_mat = gene_mat.transpose()

    mat_per_label = get_samples(gene_mat, classes, {"ref": 0, "sample": 1})
    ref, sample = mat_per_label["ref"], mat_per_label["sample"]

    ref_mean = ref.mean(axis=1)
    ref_var = ref.var(axis=1)

    sample_mean = sample.mean(axis=1)
    sample_var = sample.var(axis=1)

    if de_method == "fc":
        fc = ref_mean / sample_mean
        output_file = os.path.join(output_dir, "fold_change.txt")
        fc.to_csv(output_file, sep="\t", header=False)
    else:
        z_score = (ref_mean - sample_mean) / np.sqrt(
            (ref_var / ref.shape[1]) + (sample_var / sample.shape[1])
        )
        output_file = os.path.join(output_dir, "zscore.txt")
        z_score.to_csv(output_file, sep="\t", header=False)
    return output_file


def run_executables(scores_file: str, drug_output_dir: str, kwargs: Kwargs):
    """
    run subgraphILP executables and save output to the disk. progress is logged in a separate file
    """

    log_file = open(kwargs.subgraphilp_logger, "a")
    subprocess.run(
        f'echo {str(datetime.now()).split(".")[0]}:{kwargs.data.drug_name}'.split(),
        stdout=log_file,
    )

    mapper_command = (
        f"./{kwargs.training.mapper_path} "
        f"-s {scores_file} "
        f"-a {drug_output_dir}/score_orig.na "
        f"-o {drug_output_dir}/score.na "
        f"-i {kwargs.data.node_entrez_file} "
        f"-c {kwargs.data.aggregate_file} "
        f"-f max "
        f"-p min"
    )
    subprocess.run(mapper_command.split(), stdout=log_file)

    if kwargs.training.target_root_node:
        drug = kwargs.data.drug_name.split("___")[0]
        drug_targets = kwargs.data.processed_files.drugs_targets
        if drug in drug_targets.keys():
            for target_node in drug_targets[drug]:
                subprocess.run(
                    f'echo "\ntarget node ID: {target_node}"'.split(), stdout=log_file
                )
                node_dir = os.path.join(drug_output_dir, str(target_node))
                os.makedirs(node_dir, exist_ok=True)
                comp_command = (
                    f"./{kwargs.training.comp_path} "
                    f"-s {drug_output_dir}/score.na "
                    f"-e {kwargs.data.kegg_hsa_file} "
                    f"-re {node_dir}/network.%k.sif "
                    f"-k {kwargs.training.num_knodes} "
                    f"-sr {target_node}"
                )
                subprocess.run(comp_command.split(), stdout=log_file)
    else:
        comp_command = (
            f"./{kwargs.training.comp_path} "
            f"-s {drug_output_dir}/score.na "
            f"-e {kwargs.data.kegg_hsa_file} "
            f"-re {drug_output_dir}/network.%k.sif "
            f"-k {kwargs.training.num_knodes}"
        )
        subprocess.run(comp_command.split(), stdout=log_file)
    log_file.close()


def process_output(nets_dir: str, features: List, kwargs: Kwargs):
    """
    process subgraphILP output files
    nets_dir: str of a path to the subgraphilp output files
    features: the original set of features (genes)
    """
    subgraph_files = [
        os.path.join(nets_dir, file)
        for file in os.listdir(nets_dir)
        if file.endswith(".sif")
    ]

    # combining subgraphs by collecting all the genes existing in all the reported subgraphs
    all_entrez = all_no_aggs = all_nodes = set()
    for file in subgraph_files:
        subgraph = pd.read_csv(file, sep="\t", names=["source", "edge", "sink"])
        no_aggs, nodes, entrez = map_nodes_to_entrez(
            subgraph,
            kwargs.data.processed_files.node_entrez,
            kwargs.data.processed_files.aggregates,
        )
        all_entrez.update(entrez)
        all_no_aggs.update(no_aggs)
        all_nodes.update(nodes)

    if isinstance(features[0], str):
        # features are forced to be strings after being passed to RF.
        all_entrez = [str(feature) for feature in all_entrez]
    selected_features = sorted(list(set(features).intersection(all_entrez)))

    # The gene matrix is given as gene symbols, while subgraphilp maps only to entrez IDs.
    # The gene matrix is filtered to include only genes with corresponding entrez ids available in
    # "CTD_genes_pathways.csv". Then, this filtered list is further filtered to the entrez present in "kegg_hsa.sif".
    # it's worth noticing that some kegg genes are not recognized by the ctd file, and some kegg genes recognized by ctd
    # are not present in the original matrix. Therefore, the expression is not fetched for these genes.
    # The below stores these mappings and outputs it to a json file for further inspection.
    entrez_symbols = kwargs.data.processed_files.entrez_symbols
    entrez_in_ctd = set(entrez_symbols["GeneID"]).intersection(all_entrez)
    present_entrez_symbols = entrez_symbols[
        entrez_symbols["GeneID"].isin(entrez_in_ctd)
    ]
    selection_info = {
        "nodes_only": all_no_aggs,
        "nodes_with_aggregates": all_nodes,
        "nodes_entrez": all_entrez,
        "matched_ctd_genes": present_entrez_symbols,
        "present_in_gene_matrix": selected_features,
    }
    output_selection = os.path.join(nets_dir, "feature_selection_info.json")
    with open(output_selection, "w") as selection:
        selection.write(json.dumps(selection_info, indent=2, cls=NewJsonEncoder))
    return selected_features


def extract_features(nets_dir: str, features: list, kwargs: Kwargs):
    if kwargs.training.target_root_node:
        nodes_folders = [
            os.path.join(nets_dir, folder)
            for folder in os.listdir(nets_dir)
            if os.path.isdir(os.path.join(nets_dir, folder))
        ]
        if len(nodes_folders) > 0:
            selected_features = set()
            for node_folder in nodes_folders:
                node_features = process_output(node_folder, features, kwargs)

                selected_features.update(node_features)
            selected_features = list(selected_features)
        else:
            selected_features = None  # TODO: check if this case would ever happen?
    else:
        selected_features = process_output(nets_dir, features, kwargs)

    return selected_features


def subgraphilp(
    train_features: pd.DataFrame,
    train_classes: pd.Series,
    kwargs: Kwargs,
    tree_idx: Union[int, None] = None,
):
    """
    perform subgraphILP feature selection
    train_features = pd.DataFrame(..., columns=[genes: List[Any], index=[cell_lines: List[int]])
    train_classes = pd.Series(..., columns=["labels"], index=[cell_lines: List[int]])
    k = the index of the current cross validation
    tree_idx = the index of the current tree in the RF
    return:
        selected_features List of selected genes
    """
    if kwargs.training.cv_idx is not None:
        drug_output_dir = os.path.join(
            kwargs.intermediate_output,
            kwargs.data.drug_name,
            f"params_comb_{kwargs.training.gcv_idx}",
            f"subgraphilp_cv_{kwargs.training.cv_idx}",
        )
    else:
        drug_output_dir = os.path.join(
            kwargs.intermediate_output,
            kwargs.data.drug_name,
            f"subgraphilp_best_model",
        )
    if tree_idx is not None:
        drug_output_dir = os.path.join(drug_output_dir, f"tree_{tree_idx}")
    os.makedirs(drug_output_dir, exist_ok=True)

    de_file = differential_expression(
        train_features,
        train_classes,
        drug_output_dir,
        de_method=kwargs.training.de_method,
    )

    run_executables(de_file, drug_output_dir, kwargs)

    selected_features = extract_features(
        drug_output_dir, train_features.columns.to_list(), kwargs
    )
    return selected_features
