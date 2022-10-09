import os
from functools import cached_property
from typing import Any, List, Optional, Union

from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid

from manager.data.files_preprocessing import ProcessedFiles


class TrainingConfig(BaseModel):
    class Config:
        """pydantic breaks with cached_property decorator. hence, modifying the configuration"""

        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    # subgraphilp parameters
    mapper_path: Optional[str]  # path to mapper.exe
    comp_path: Optional[str]  # path to compOpt.exe
    de_method: str  # [fc, zscore]
    num_knodes: str = "3-20"
    target_root_node: bool

    # RF parameters
    max_features: Optional[Union[str, List[Any]]]  # TODO: change
    min_samples_leaf: Union[int, List[int]] = 20
    n_estimators: Optional[Union[int, List[int]]] = 500
    n_jobs: Optional[int] = 3
    random_state: Optional[int] = 42
    bias_rf: bool
    bias_pct: Optional[float] = 1
    output_trees_info: bool = False
    sauron: bool

    @cached_property
    def parameters_grid(self) -> dict:
        if not self.max_features:
            if self.regression:
                self.max_features = ["sqrt"]
            else:
                self.max_features = [None]  # [0.3]  # TODO: change
        elif isinstance(self.max_features, str):
            self.max_features = [self.max_features]

        if isinstance(self.min_samples_leaf, int):
            self.min_samples_leaf = [self.min_samples_leaf]
        elif len(self.min_samples_leaf) == 3:
            range_ = self.min_samples_leaf
            self.min_samples_leaf = list(range(range_[0], range_[1], range_[2]))

        if isinstance(self.n_estimators, int):
            self.n_estimators = [self.n_estimators]
        elif len(self.n_estimators) == 3:
            range_ = self.n_estimators
            self.n_estimators = list(range(range_[0], range_[1], range_[2]))

        rf_hyperparameters = {
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,  # restrict depth
            "n_estimators": self.n_estimators,
            "random_state": [self.random_state],
            "n_jobs": [self.n_jobs],
        }
        grid = {
            idx: params
            for idx, params in enumerate(list(ParameterGrid(rf_hyperparameters)))
        }
        return grid

    grid_search: bool = True
    drugs_n_jobs: int = 1

    # cross validation parameters
    num_kfold: int = 5
    cv_idx: Optional[int]
    gcv_idx: Optional[int]

    # correlation parameters
    corr_thresh: float = 0.3
    corr_thresh_sub = 0.05  # the percentage of features to select if no features were found at the given threshold

    # random model parameters
    num_random_samples: int = 10

    # experimentation parameters
    weight_samples: bool
    simple_weight: Optional[bool]
    regression: bool
    cell_lines_thresh: int = 700  # min number of cell lines per drug


class ModelConfig(BaseModel):
    class Config:
        """pydantic breaks with cached_property decorator. hence, modifying the configuration"""

        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    models: Union[str, List[str]] = [
        "subgraphilp",
        "original",
        "random",
        "corr_thresh",
        "corr_num",
    ]
    current_model: Optional[str]

    @cached_property
    def model_names(self) -> List[str]:
        if isinstance(self.models, str):
            return [self.models]
        else:
            return self.models


class DataConfing(BaseModel):
    class Config:
        """pydantic breaks with cached_property decorator. hence, modifying the configuration"""

        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    gene_matrix_file: str  # path to tsv file

    # dynamic fields
    drug_name: Optional[str]
    drug_threshold: Optional[float]
    output_num_feature: Optional[bool] = True
    num_features_file: Optional[str]
    drug_subset: Optional[Union[int, List[str]]]
    acc_subset: Optional[list]

    # --- GDSC files ---
    ic50_gdsc_file: str  # path to tsv file
    discretized_gdsc_file: str  # path to tsv file
    thresholds_file: str  # path to tsv file for thresholds to discretize ic_50 values
    drugs_targets_file: Optional[
        str
    ]  # path to tsv file containing drug name and corresponding comma-separated targets

    # --- KEGG files ---
    kegg_hsa_file: str  # path to the kegg network file
    node_entrez_file: str  # path to the file mapping kegg network nodes to Entrez ids
    aggregate_file: str  # path to the file identifying protein families

    # --- CTD files ---
    ctd_file: str  # path to the ctd file

    @cached_property
    def processed_files(self) -> ProcessedFiles:
        return ProcessedFiles(self)


class Kwargs(BaseModel):
    class Config:
        """pydantic breaks with cached_property decorator. hence, modifying the configuration"""

        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    model: ModelConfig
    training: TrainingConfig
    data: DataConfing
    output_dir: str  # folder path to store model results
    overwrite_results: Optional[bool]
    output_selected_drugs: bool = True  # TODO: can be removed when done

    @cached_property
    def method(self):
        if self.training.regression:
            return "regression"
        else:
            return "classification"

    @cached_property
    def weight(self):
        if self.training.weight_samples:
            return "_weighted"
        else:
            return "_notweighted"

    @cached_property
    def bias(self):
        if self.training.bias_rf:
            return "_biased"
        else:
            return ""

    @cached_property
    def target(self):
        if self.training.target_root_node:
            return "_targeted"
        else:
            return ""

    @cached_property
    def sauron_rf(self):
        if self.training.sauron:
            return "_sauron"
        else:
            return ""

    @cached_property
    def matrices_output_dir(self) -> str:
        mat_dir = os.path.join(
            self.output_dir, f"drugs_per_cells_gt_{self.training.cell_lines_thresh}"
        )
        os.makedirs(mat_dir, exist_ok=True)
        return mat_dir

    @cached_property
    def from_disk(self) -> bool:
        if len(os.listdir(self.matrices_output_dir)) > 0:
            return True
        else:
            return False

    @cached_property
    def results_dir(self) -> str:
        if (
            len(self.model.model_names) == 1
            and self.model.model_names[0] == "subgraphilp"
        ):
            folder = f"{self.training.de_method}_"
        else:
            folder = ""
        new_dir = os.path.join(
            self.output_dir,
            f"{folder}{self.method}{self.weight}{self.bias}{self.target}"
            f"{self.sauron_rf}_"
            f"gt_{self.training.cell_lines_thresh}",
        )
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    @cached_property
    def results_doc(self) -> str:
        if self.training.grid_search:
            return os.path.join(
                self.results_dir,
                f"grid_search_cross_validation.jsonl",
            )
        else:
            return os.path.join(
                self.results_dir,
                f"final_model.jsonl",
            )

    @cached_property
    def subgraphilp_logger(self) -> str:
        return os.path.join(self.results_dir, "subgraphilp_log.txt")

    @cached_property
    def subgraphilp_num_features_output_file(self) -> str:
        doc = os.path.join(self.results_dir, "subgraph_num_features.txt")
        if not os.path.isfile(doc):
            with open(doc, "w") as temp:
                temp.write("drug_name,num_features\n")
        return doc

    @cached_property
    def intermediate_output(self) -> str:
        return os.path.join(self.results_dir, "intermediate_output")
