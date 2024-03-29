# Biasing Random Forests by Incorporation of a Network-Based ILP Feature Selection

## Requirements

* Python version >= 3.8
* Packages [pydantic, sklearn, pandas, numpy, matplotlib]
    * if you have `poetry`, you can install the environment by running `poetry install` in the terminal

## Project Structure
* The `manager` folder contains the code for the project.
  * `manager/data` contains the code used for data processing (e.g., files pre-processing, data splitting, data filtering, ...etc.)
  * `manger/models` contains the code to implement the random forest modifications
    * The main project ideas are implemented in `manager/models/utils/biased_random_forest.py`
  * `manger/training` contains the code for training (e.g., parameter grid search, cross validation, ...etc.)
  * `manager/visualization/plot_results_analysis` contains the code to generate analysis plots (e.g., performance, runtime, splits, ...etc.)
  * `manager/visualization/databases_distribution` contains the codes used to generate stats/figures outside the results folder
  * `manager/train.py` is the script called for training with and without grid search cross validation
  * `manager/config.py` includes the different parameters used for training

* The `data` folder contains all data used for the project (except the gene matrix file for size constraints, which can be found [here](https://drive.google.com/file/d/1iEfgMjnRQ6CEBfwNHcKwVGJwys0t_qzd/view?usp=sharing))

## Model Training
* `config.json` is an example for parameters file to be passed
  * Other parameters can be found in `manager/config.py`
* To train models with the specified parameters (or parameters grid) in `config.json` and/or test it on best performing parameters (in case of parameters grid)
  * `$ python3 cli.py config.json`
* The results are exported to a new folder `results`
* The results can be plotted using `manger/plots/main.py` where the figures will be exported to `figures` folder

