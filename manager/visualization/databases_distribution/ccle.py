"""
Data file obtained from the visualization data of the publication at the following link (tables 1 and 11)
https://www.nature.com/articles/nature11003#Sec3

4 columns were selected, "cell line primary name", "CCLE tumor type", "Compound", and "IC50"
"""

import pandas as pd


def process_ccle(print_analysis=False):
    print("Processing CCLE")
    cl_drugs = pd.read_csv("ccle_drug_response.csv", encoding="ISO-8859-1").dropna()
    cl_drugs.columns = ["cell_line", "drug", "response"]
    cl_per_drugs = cl_drugs.groupby("drug").count()["cell_line"]
    cl_per_drugs.to_csv("cl_per_drug_ccle.csv")
    if print_analysis:
        print("number of drugs = ", len(cl_per_drugs))
        print("average cell lines per drugs = ", cl_per_drugs.mean())
        print("standard deviation cell lines per drugs = ", cl_per_drugs.std())

    cl_meta = pd.read_csv("ccle_cell_lines_metadata.csv")
    cl_meta.columns = ["cell_line", "tumor_type"]
    cl_per_type = cl_meta.groupby("tumor_type").count()["cell_line"]
    cl_per_type.to_csv("cl_per_type_ccle.csv")
    if print_analysis:
        print("\nnumber of cell lines = ", len(cl_meta))
        print("number of cancer types = ", len(cl_per_type))
        print("average cell lines per cancer type =", cl_per_type.mean())
        print("standard deviation cell lines per cancer type =", cl_per_type.std())
