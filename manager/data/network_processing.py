import copy

import pandas as pd


def fetch_target_nodes(drugs_targets, node_entrez, aggregates, entrez_symbols, sif):
    """
    Given a txt file of drug name and corresponding GeneSymbols, this function gets the matching entrez GeneIDs, and
    their correspondence node ids. It reports a node id only if it exists in the network file.
    """
    drug_target_node = {}
    network_nodes = set(sif["source"].to_list()).intersection(sif["sink"].to_list())
    for drug, targets in drugs_targets.items():
        drug_targets = [target for target in targets.split(",")]
        for target in drug_targets:
            if target in entrez_symbols["GeneSymbol"].to_list():
                gene_id_df = entrez_symbols[entrez_symbols["GeneSymbol"] == target]
                gene_id = gene_id_df["GeneID"].to_list()[0]
                if gene_id in node_entrez["GeneID"].to_list():
                    gene_node_df = node_entrez[node_entrez["GeneID"] == gene_id]
                    gene_node = gene_node_df["node"].to_list()[0]
                    if gene_node not in network_nodes:
                        # continue
                        for _, aggregate_row in aggregates.iterrows():
                            if gene_node in aggregate_row["aggregates"]:
                                gene_node = aggregate_row["node"]
                                break
                    if drug in drug_target_node:
                        drug_target_node[drug].append(gene_node)
                    else:
                        drug_target_node[drug] = [gene_node]
    return drug_target_node


def mapped_nodes(node_entrez_file, aggregate_file):
    """
    Given the two files of "kegg_hsa_Entrez-Gene" and "kegg_hsa_aggregates", the function parses the files and flatten
    the grouped information from the aggregated nodes in the "kegg_hsa_aggregates" file
    """
    mapping = pd.read_csv(
        node_entrez_file,
        sep="\t=\t",
        names=["node", "GeneID"],
        skiprows=1,
        engine="python",
    )
    aggregates = pd.read_csv(
        aggregate_file, sep="\t=\t", names=["node", "aggregates"], engine="python"
    )

    # since aggregate groups are captured as strings, this will convert all gene groups to list of integers to be
    # comparable with the later analysis
    to_replace = {
        "[": "",
        "{": "",
        "}": "",
        "]": "",
        ":": ",",
    }  # no difference between families and complexes
    for old, new in to_replace.items():
        aggregates["aggregates"] = aggregates["aggregates"].str.replace(
            old, new, regex=False
        )
    aggregates["aggregates"] = (
        aggregates["aggregates"].str.split(",").apply(lambda x: [int(i) for i in x])
    )
    return mapping, aggregates


def map_nodes_to_entrez(subgraph, mapping, aggregates):
    """
    Converts node IDs of the subgraph networks (or provided list of network node IDs), to entrez IDs.
    Each node is checked if it's matched to an aggregated node, if matched, the list of aggregates are parsed to
    the set of nodes (as these aggregates are the actual gene_corresponding_ids), instead of the original aggregated ID.
    The final set of nodes is checked against the node-Entrez mapping and entrez IDs are returned instead.

    @param subgraph: pandas dataframe with columns=['source', 'edge', 'sink']
    @param mapping: pandas dataframe with columns=['node', 'GeneID']
    @param aggregates: pandas dataframe with columns=['node', 'aggregates']; 'aggregates' consists of lists of node IDs
    @return:
    """
    if isinstance(subgraph, list):
        nodes = set(subgraph)
    else:
        nodes = set(subgraph["source"].to_list())
        nodes.update(subgraph["sink"].to_list())
    no_aggs = copy.deepcopy(nodes)

    aggregated = aggregates[aggregates["node"].isin(nodes)]
    for group in aggregated["aggregates"].to_list():
        nodes.update(group)

    for node in aggregated["node"].to_list():
        nodes.remove(node)

    mapped_df = mapping[mapping["node"].isin(nodes)]
    mapped = mapped_df["GeneID"].to_list()
    return no_aggs, nodes, mapped
