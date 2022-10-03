import pandas as pd
from manager.data.network_processing import mapped_nodes

kegg_entrez, kegg_aggregates = mapped_nodes(
    "kegg_hsa.Entrez-Gene.na", "kegg_hsa.aggregate.na"
)
kegg_enz = kegg_entrez["node"]

aggregates_parsed_nodes = set()
for agg in kegg_aggregates["aggregates"]:
    aggregates_parsed_nodes.update(agg)
print(
    f"kegg_entrez intersection with aggregates super nodes = {len(set(kegg_enz).intersection(kegg_aggregates['node']))}\n"
    f"kegg_entrez intersection with aggregates parsed nodes = {len(set(kegg_enz).intersection(aggregates_parsed_nodes))}"
)

kegg_hsa = pd.read_csv("kegg_hsa.sif", sep="\t", names=["source", "edge", "sink"])
kegg_nodes = set(kegg_hsa["source"])
kegg_nodes.intersection_update(kegg_hsa["sink"])

print(
    f"number of nodes in kegg_hsa = {len(kegg_nodes)}\n"
    f"number of nodes with matched entrez id = {len(kegg_enz)}\n"
    f"Number of parsed nodes in aggregates = {len(aggregates_parsed_nodes)}"
)

subgraphilp_mapping = pd.read_csv(
    "score.na", sep="\t=\t", names=["node", "score"], skiprows=1, engine="python"
)

print(
    f"subgraphILP mapping to kegg_entrez = {len(set(subgraphilp_mapping['node']).intersection(kegg_entrez['node']))}\n"
    f"subgraphILP mapping to aggregates super nodes = {len(set(subgraphilp_mapping['node']).intersection(kegg_aggregates['node']))}\n"
    f"subgraphILP mapping to aggregates parsed nodes = {len(set(subgraphilp_mapping['node']).intersection(aggregates_parsed_nodes))}"
)
ctd_entrez = pd.read_csv("CTD_genes_pathways.csv")[
    ["GeneID", "GeneSymbol"]
].drop_duplicates()
