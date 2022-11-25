"""
The data was collected from the publication at the following link (NIHMS711523-supplement-4.xlsx)
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631646/

cell lines metadata was collected from S2 sheet ("index_ccl" and "ccle_primary_histology" columns were stored as csv file)
drugs info was collected from S1 sheet ("index_cpd" and "index_ccl" columns were stored as csv file)
"""

import pandas as pd


def process_ctrpv2(print_analysis=False):
    print("Processing CTRPv2")
    drug_cell_lines = pd.read_csv("ctrpv2_drug_cell_lines.csv")
    cl_per_drugs = drug_cell_lines.groupby("index_cpd").count()["index_ccl"]
    cl_per_drugs.to_csv("cl_per_drug_ctrpv2.csv")
    if print_analysis:
        print("number of drugs = ", len(cl_per_drugs))
        print("average cell lines per drugs = ", cl_per_drugs.mean())
        print("standard deviation cell lines per drugs = ", cl_per_drugs.std())

    cell_metadata = pd.read_csv("ctrpv2_cell_lines_meta_data.csv")
    cl_per_type = cell_metadata.groupby("ccle_primary_histology").count()["index_ccl"]
    cl_per_type.to_csv("cl_per_type_ctrpv2.csv")
    if print_analysis:
        print("\nnumber of cell lines = ", len(cell_metadata))
        print("number of cancer types = ", len(cl_per_type))
        print("average cell lines per cancer type =", cl_per_type.mean())
        print("standard deviation cell lines per cancer type =", cl_per_type.std())
