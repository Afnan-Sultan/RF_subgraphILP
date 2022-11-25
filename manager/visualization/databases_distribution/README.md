# Processing and Plotting pre-clinical cancer models databases metadata

* This folder contains metadata files fetched from different databases. Each `{database}.py` script contains information at the header explaining how each `*.csv` file was obtained.
* `plot_databases_distribution` calls every `{database}.py` script, then plots the results as boxplot. The outputs of `{database}.py` and the plot's figure are saved to the current directory.
* The saved figure is the one used in the master thesis (Figure 1.2)