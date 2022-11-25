"""
Data file obtained from the following link
https://discover.nci.nih.gov/cellminer/loadDownload.do
by choosing "Download Raw Data Set --> Compound activity: DTP NCI-60 --> Get Raw Data Set"

last online access: 20.09.2022

For file pre-processing:
1- "NSC #" column was used as drug identifier while ignoring all other metadata
2- the relevant rows of drug identifier and cell lines were stored as csv file
"""

import statistics

import pandas as pd


def process_nci60(print_analysis=False):
    print("Processing NCI60")
    nci60 = pd.read_csv("nci60_database.csv", encoding="ISO-8859-1")
    drug_cell_lines = {}
    for _, row in nci60.iterrows():
        drug_name = row["NSC #"]
        if drug_name in drug_cell_lines:
            drug_cell_lines[drug_name].update(
                row.drop(["NSC #"]).dropna().index.to_list()
            )
        else:
            drug_cell_lines[drug_name] = set(
                row.drop(["NSC #"]).dropna().index.to_list()
            )
    cl_per_drugs = {
        drug: len(cell_lines) for drug, cell_lines in drug_cell_lines.items()
    }
    pd.DataFrame(cl_per_drugs, index=["cell_lines_count"]).transpose().to_csv(
        "cl_per_drug_nci60.csv"
    )
    if print_analysis:
        print("number of drugs = ", len(cl_per_drugs))
        print("average cell lines per drugs = ", statistics.mean(cl_per_drugs.values()))
        print(
            "standard deviation cell lines per drugs = ",
            statistics.stdev(cl_per_drugs.values()),
        )

    cell_lines = nci60.columns.to_list()[1:]
    type_cell_line = {}
    for cl in cell_lines:
        cl = cl.split(":")
        cl_type = cl[0]
        cl_subtype = cl[1:]
        if cl_type in type_cell_line:
            type_cell_line[cl_type].append(cl_subtype)
        else:
            type_cell_line[cl_type] = [cl_subtype]
    cl_per_type = {
        cl_type: len(cl_subtypes) for cl_type, cl_subtypes in type_cell_line.items()
    }
    pd.DataFrame(cl_per_type, index=["num_cell_lines"]).transpose().to_csv(
        "cl_per_type_nci60.csv"
    )
    if print_analysis:
        print("\nnumber of cell lines = ", len(cl_per_type))
        print(
            "average cell lines per cancer type =",
            statistics.mean(cl_per_type.values()),
        )
        print(
            "standard deviation cell lines per cancer type =",
            statistics.stdev(cl_per_type.values()),
        )
