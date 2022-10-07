import json
import os.path
import subprocess
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from manager.config import Kwargs
from manager.data.network_processing import map_nodes_to_entrez
from manager.utils import NewJsonEncoder


def get_samples(gene_mat, classes, label):
    label_cls = classes[classes.values == label].index.to_list()
    to_include = []
    for idx, cl in enumerate(gene_mat.columns.to_list()):
        if cl in label_cls:
            to_include.append(idx)
    label_mat = gene_mat.iloc[:, to_include]
    return label_mat


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

    ref = get_samples(gene_mat, classes, 0)
    ref_mean = ref.mean(axis=1)
    ref_var = ref.var(axis=1)

    sample = get_samples(gene_mat, classes, 1)
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


def run_subgraphilp_executables(scores_file: str, drug_output_dir: str, kwargs: Kwargs):
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


def combine_subgraphs(
    subgraph_files: List[str], mapping: pd.DataFrame, aggregates: pd.DataFrame
):
    """
    collect all the genes existing in all the reported subgraphs

    @param subgraph_files: list, files paths
    @param mapping: pd.DataFrame(..., columns=['node', 'GeneID'])
    @param aggregates: pd.DataFrame(..., columns=['node', 'GeneID'])
    @return: set of all mapped entrezIDs in the networks
    """
    all_entrez = set()
    all_no_aggs = set()
    all_nodes = set()
    for file in subgraph_files:
        subgraph = pd.read_csv(file, sep="\t", names=["source", "edge", "sink"])
        no_aggs, nodes, entrez = map_nodes_to_entrez(subgraph, mapping, aggregates)
        all_entrez.update(entrez)
        all_no_aggs.update(no_aggs)
        all_nodes.update(nodes)
    return all_no_aggs, all_nodes, all_entrez


def process_subgraphilp_output(nets_dir: str, features: List, kwargs: Kwargs):
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
    all_no_aggs, all_nodes, all_entrez = combine_subgraphs(
        subgraph_files,
        kwargs.data.processed_files.node_entrez,
        kwargs.data.processed_files.aggregates,
    )

    entrez_symbols = kwargs.data.processed_files.entrez_symbols
    entrez_in_ctd = set(entrez_symbols["GeneID"]).intersection(all_entrez)
    present_entrez_symbols = entrez_symbols[
        entrez_symbols["GeneID"].isin(entrez_in_ctd)
    ]  # ['GeneSymbol'].to_list()

    if isinstance(features[0], str):
        all_entrez = [str(feature) for feature in all_entrez]
    assert type(features[0]) == type(list(all_entrez)[0])
    selected_features = sorted(list(set(features).intersection(all_entrez)))

    # The gene matrix is given as gene symbols, while the subgraphilp maps only to entrez IDs.
    # The gene matrix is filtered to include only genes with corresponding entrez ids available in
    # "CTD_genes_pathways.csv". Then, this filtered list is further filtered to the entrez present in "kegg_hsa.sif".
    # it's worth noticing that some kegg genes are not recognized by the ctd file, and some kegg genes recognized by ctd
    # are not present in the original matrix. Therefore, the expression is not fetched for these genes.
    # The below dict stores these mappings and outputs it to a json file for further inspection.
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


def subgraphilp_features(nets_dir: str, features: list, kwargs: Kwargs):
    if kwargs.training.target_root_node:
        nodes_folders = [
            os.path.join(nets_dir, folder)
            for folder in os.listdir(nets_dir)
            if os.path.isdir(os.path.join(nets_dir, folder))
        ]
        if len(nodes_folders) > 0:
            selected_features = set()
            for node_folder in nodes_folders:
                node_selection_info, node_features = process_subgraphilp_output(
                    node_folder, features, kwargs
                )

                selected_features.update(node_features)
            selected_features = list(selected_features)
        else:
            selected_features = None  # TODO: check if this case would ever happen?
    else:
        selected_features = process_subgraphilp_output(nets_dir, features, kwargs)

    return selected_features
