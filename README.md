# Biasing Random Forests by Incorporation of a Network-Based ILP Feature Selection

## Requirements

* Python version >= 3.8
* Packages [pydantic, sklearn, pandas, numpy, matplotlib]
    * if you have `poetry`, you can install the environment by running `poetry install` in the terminal

## Project Structure
* The `manager` folder contains the code for the project.
  * `manager/data` contains the code used for data processing (e.g., files pre-processing, data splitting, data filtering, ...etc.)
  * `manger/models` contains the code to implement the different models (e.g., subgraphILP, random, ...etc.)
  * `manger/training` contains the code for training (e.g., parameter grid search, cross validation, ...etc.)
  * `manager/plots` contains the code to generate analysis plots (e.g., performance, runtime, splits, ...etc.)
  * `manager/scoring_metrics` contains the code used for evaluation
  * `manager/supplementary` contains the codes used to generate stats/figure outside the results folder
  * `manager/train.py` is the script called for training with grid search cross validation
  * `manager/train_final.py` is used to train a single model with no cross validation
  * `manager/config.py` includes the different parameters used for training
  

* The `data` folder contains all data used for the project (except the gene matrix for size constraints, which can be found at ??)

## Model Training
* `config.json` is an example for parameters file to be passed
  * Other parameters can be found in `manager/config.py`
* To cross validate the model with the specified parameters (or parameters grid) in `config.json` and test it on best performing parameters (in case of parameters grid)
  * `$ python3 cli.py train config.json`
* To train and test the model with fixed parameters list directly
  * `$ python3 cli.py test config.json`
* The results are exported to a new file `results`
* The results can be plotted using `manger/plots/main.py` where the figures will be exported to `figures` folder

