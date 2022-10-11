"""
This file extends sklearn RandomForestClassifier/Regressor.
1. each model (classifier/regressor) is extended with new class properties to be used for further processing
2. {model}.fit is extended to implement our new fitting techniques
    2.1 _parallel_build_trees is where the main changes occur.
        * Each tree is biased if bias_rf is selected
        * Each tree regressor leaf is assigned a class if sauron is selected
3. new property is introduced, biased_feature_importance, to account for the missing features in each tree in case
   bias_rf is selected
4. {model}.predict is extended to predict response based on leaf_class_assignment if sauron is selected
"""

import json
import os.path
import threading
from typing import Union
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import pandas as pd
from joblib import Parallel
from manager.config import Kwargs
from manager.models.biased_decision_tree import (BiasedDecisionTreeClassifier,
                                                 BiasedDecisionTreeRegressor)
from manager.training.feature_selection import feature_selection
from manager.utils import NewJsonEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (MAX_INT, ForestClassifier,
                                      ForestRegressor, _accumulate_prediction,
                                      _generate_sample_indices,
                                      _get_n_samples_bootstrap)
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._classes import DOUBLE, DTYPE
from sklearn.tree._tree import issparse
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight, check_is_fitted


class BiasedRandomForestRegressor(ForestRegressor):
    def __init__(
        self,
        kwargs: Kwargs,
        train_classes,
        train_scores,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=BiasedDecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state,
            ),
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.kwargs = kwargs
        self.train_classes = train_classes
        self.train_scores = train_scores

    def fit(self, X, y, sample_weight=None):
        return biased_fit(
            self,
            X,
            y,
            sample_weight=sample_weight,
        )

    def predict(self, X):
        """
        same function as the model's original function. However, in case of bias_rf, the dataset needs to be subsetted
        for each tree to include only the features used during fitting.

        the tree.predict function accepts only ndarrays with the assumption that the ndarray contains the same number
        of features as used in tree.fit
        """
        if self.kwargs.training.sauron:
            return predict_sauron(self, X)

        check_is_fitted(self)

        # Check data
        X = self._validate_X_predict(X)

        # data validation converts X to ndarray, but we need feature names to subset for each tree in case of bias_rf
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(
                e.predict,
                # tree.predict accepts ndarrays. So, the dataset is subsetted to include only features of the tree
                # before converting to ndarray
                np.array(X_df.loc[:, e.feature_names_in_]),
                [y_hat],
                lock,
            )
            for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat

    @property
    def biased_feature_importance(self):
        return biased_feature_importances_(self)


class BiasedRandomForestClassifier(ForestClassifier):
    def __init__(
        self,
        kwargs: Kwargs,
        train_classes,
        train_scores,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=BiasedDecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state,
            ),
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.kwargs = kwargs
        self.train_classes = train_classes
        self.train_scores = train_scores

    def fit(self, X, y, sample_weight=None):
        return biased_fit(self, X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        """
        same function as the model's original function. However, in case of bias_rf, the dataset needs to be subsetted
        for each tree to include only the features used during fitting.

        the tree.predict function accepts only ndarrays with the assumption that the ndarray contains the same number
        of features as used in tree.fit
        """
        check_is_fitted(self)

        # Check data
        X = self._validate_X_predict(X)

        # data validation converts X to ndarray, but we need feature names to subset for each tree in case of bias_rf
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()

        # define the input dataframe with the features' names, to subset it properly for each tree
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(
                e.predict_proba,
                # pass only the features' subset that each tree was fitted on
                np.array(X_df.loc[:, e.feature_names_in_]),
                all_proba,
                lock,
            )
            for e in self.estimators_
        )

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    @property
    def biased_feature_importance(self):
        return biased_feature_importances_(self)


def output_tree_info(
    bootstrapped_samples, biased_features, importance, tree_idx, kwargs
):
    """
    This is meant to be used for best model only
    """
    if kwargs.training.gcv_idx is None and kwargs.training.cv_idx is None:
        out_path = os.path.join(
            kwargs.intermediate_output, kwargs.data.drug_name, "rf_trees_info"
        )
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, f"{kwargs.model.current_model}.json")
        with open(out_file, "a") as tree_output:
            tree_output.write(
                json.dumps(
                    {
                        tree_idx: {
                            "cell_lines": bootstrapped_samples,
                            "features": biased_features,
                            "features_importance": importance,
                        }
                    },
                    indent=2,
                    cls=NewJsonEncoder,
                )
            )
            tree_output.write("\n")


def leaf_class_assignment(tree, bootstrapped_X, X_classes, bootstrapped_X_weights):
    """
    assign a class for the RF_Regressor leafs after fitting, based on pre-defined samples' classification

    X_classes contains classes for all cell_lines to avoid multiple retrieval of classes if bootstrapped_X was used
    """

    # identify the leaf node for each sample
    samples_assignment = tree.apply(bootstrapped_X.loc[:, tree.feature_names_in_])

    # storage purposes - store the ids/classes of cell lines falling into a leaf
    leaf_samples, leaf_classes = {}, {}
    cell_lines = bootstrapped_X.index.to_list()

    # store the leaf information, samples true ids (i.e., cell lines IDs) and samples true classes (i.e., 0/1)
    for sample_idx, leaf in enumerate(samples_assignment):
        cell_line = cell_lines[sample_idx]
        if leaf in leaf_samples:
            leaf_samples[leaf].append(cell_line)
            leaf_classes[leaf].append(
                [X_classes.loc[cell_line], bootstrapped_X_weights.loc[cell_line]]
            )
        else:
            leaf_samples[leaf] = [cell_line]
            leaf_classes[leaf] = [
                [X_classes.loc[cell_line], bootstrapped_X_weights.loc[cell_line]]
            ]
    leaf_classes_dfs = {
        leaf: pd.DataFrame(
            assignment, columns=["label", "weight"], index=leaf_samples[leaf]
        )
        for leaf, assignment in leaf_classes.items()
    }

    # fetch the corresponding class for each leaf by majority vote over the `leaf_classes`
    leaf_assignments = {
        leaf: 1
        if assignment_df[assignment_df["label"] == 1]["weight"].sum()
        >= assignment_df[assignment_df["label"] == 0]["weight"].sum()
        else 0
        for leaf, assignment_df in leaf_classes_dfs.items()
    }
    return leaf_classes_dfs, leaf_assignments


def bias_feature_importance(tree, features_name):
    """
    store the feature_importance for the features of the tree and set the importance for remaining features to zero
    """
    importance = np.zeros(
        len(features_name)
    )  # set to the size of the original features
    tree_features_indices = [
        features_name.tolist().index(
            feature
        )  # get the index of the current feature to assign importance
        for feature in tree.feature_names_in_
    ]
    importance[tree_features_indices] = tree.feature_importances_
    return importance


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    kwargs: Kwargs,
    train_classes,
    train_scores,
    features_name,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    """
    Private function used to fit a single tree in parallel.

    Copied form the sklearn main code, with interfering to add the desired changes for each tree.
    Personal changes are encapsulated by %%%%%%%%%% mark
    """
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )

        # %%%%%%%%%%
        # convert X to a dataframe to make the feature names recognized by the trees
        X = pd.DataFrame(X, columns=features_name, index=train_classes.index.to_list())

        # identify the bootstrapped matrix and corresponding scores/classes
        bootstrapped_X = X.iloc[indices]
        tree.bootstrapped_samples = bootstrapped_X.index.to_list()  # for storage

        bootstrapped_X_scores = train_scores.iloc[indices]
        bootstrapped_X_classes = train_classes.iloc[indices]

        if sample_weight is None:
            # set the weight of each sample to 1
            unique_samples = list(set(tree.bootstrapped_samples))
            bootstrapped_X_weights = pd.Series(
                [1] * len(unique_samples), index=unique_samples
            )
        else:
            # identify the weight used for each sample in the bootstrapped sample.
            # DataFrame is used for easier duplicates drop. Duplicates are removed for lack of importance
            bootstrapped_X_weights = (
                pd.DataFrame([tree.bootstrapped_samples, sample_weight[indices]])
                .transpose()
                .drop_duplicates()
            )
            # faster to process when it's a series
            bootstrapped_X_weights = pd.Series(
                bootstrapped_X_weights[1].to_list(), index=bootstrapped_X_weights[0]
            )

        biased_features = None
        if kwargs.training.bias_rf:
            # select features of those reported from the method of interest
            to_tree = feature_selection(
                train_features=bootstrapped_X,
                train_classes=bootstrapped_X_classes,
                train_scores=bootstrapped_X_scores,
                kwargs=kwargs,
                tree_idx=tree_idx,
            )
            biased_features = to_tree["features"]
            X = X.loc[:, biased_features]
        # %%%%%%%%%%

        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        # %%%%%%%%%%
        # check_input is changed to True to force each tree to check feature names
        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=True)

        if kwargs.training.bias_rf:
            if kwargs.training.output_trees_info:
                # output tree info (bootstrapped samples, features, and features importance) for analysis
                output_tree_info(
                    bootstrapped_samples=tree.bootstrapped_samples,
                    biased_features=biased_features,
                    importance=tree.feature_importances_,
                    tree_idx=tree_idx,
                    kwargs=kwargs,
                )

            # each tree is trained on features' subset, but the RF is using all features.
            # Therefore, a value of 0 is assigned to the non-used features
            importance = bias_feature_importance(tree, features_name)
            tree.biased_feature_importance = importance
        else:
            tree.biased_feature_importance = tree.feature_importances_

        if kwargs.training.regression and kwargs.training.sauron:
            # identify leaf class if sauron is required during regression
            (
                tree.leaf_classes_dfs,
                tree.leaf_class_assignments,
            ) = leaf_class_assignment(
                tree, bootstrapped_X, train_classes, bootstrapped_X_weights
            )
        # %%%%%%%%%%
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree


def biased_fit(
    model: Union[BiasedRandomForestRegressor, BiasedRandomForestClassifier],
    X,
    y,
    sample_weight=None,
):
    """
    the same fit function as in sklearn random forest, except implementing our own _parallel_build_trees function with
    the additional features
    """

    # Validate or convert input data
    if issparse(y):
        raise ValueError("sparse multilabel-indicator for y is not supported.")
    X, y = model._validate_data(
        X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    )
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    if issparse(X):
        # Pre-sort indices to avoid that each individual tree of the
        # ensemble sorts the indices.
        X.sort_indices()

    y = np.atleast_1d(y)
    if y.ndim == 2 and y.shape[1] == 1:
        warn(
            "A column-vector y was passed when a 1d array was"
            " expected. Please change the shape of y to "
            "(n_samples,), for example using ravel().",
            DataConversionWarning,
            stacklevel=2,
        )

    if y.ndim == 1:
        # reshape is necessary to preserve the data contiguity against vs
        # [:, np.newaxis] that does not.
        y = np.reshape(y, (-1, 1))

    if model.criterion == "poisson":
        if np.any(y < 0):
            raise ValueError(
                "Some value(s) of y are negative which is "
                "not allowed for Poisson regression."
            )
        if np.sum(y) <= 0:
            raise ValueError(
                "Sum of y is not strictly positive which "
                "is necessary for Poisson regression."
            )

    model.n_outputs_ = y.shape[1]

    y, expanded_class_weight = model._validate_y_class_weight(y)

    if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
        y = np.ascontiguousarray(y, dtype=DOUBLE)

    if expanded_class_weight is not None:
        if sample_weight is not None:
            sample_weight = sample_weight * expanded_class_weight
        else:
            sample_weight = expanded_class_weight

    if not model.bootstrap and model.max_samples is not None:
        raise ValueError(
            "`max_sample` cannot be set if `bootstrap=False`. "
            "Either switch to `bootstrap=True` or set "
            "`max_sample=None`."
        )
    elif model.bootstrap:
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=model.max_samples
        )
    else:
        n_samples_bootstrap = None

    # Check parameters
    model._validate_estimator()

    if not model.bootstrap and model.oob_score:
        raise ValueError("Out of bag estimation only available if bootstrap=True")

    random_state = check_random_state(model.random_state)

    if not model.warm_start or not hasattr(model, "estimators_"):
        # Free allocated memory, if any
        model.estimators_ = []

    n_more_estimators = model.n_estimators - len(model.estimators_)

    if n_more_estimators < 0:
        raise ValueError(
            "n_estimators=%d must be larger or equal to "
            "len(estimators_)=%d when warm_start==True"
            % (model.n_estimators, len(model.estimators_))
        )

    elif n_more_estimators == 0:
        warn(
            "Warm-start fitting without increasing n_estimators does not "
            "fit new trees."
        )
    else:
        if model.warm_start and len(model.estimators_) > 0:
            # We draw from the random state to get the random state we
            # would have got if we hadn't used a warm_start.
            random_state.randint(MAX_INT, size=len(model.estimators_))

        trees = [
            model._make_estimator(append=False, random_state=random_state)
            for i in range(n_more_estimators)
        ]

        # Parallel loop: we prefer the threading backend as the Cython code
        # for fitting the trees is internally releasing the Python GIL
        # making threading more efficient than multiprocessing in
        # that case. However, for joblib 0.12+ we respect any
        # parallel_backend contexts set at a higher level,
        # since correctness does not rely on using threads.
        trees = Parallel(n_jobs=model.n_jobs, verbose=model.verbose, prefer="threads",)(
            delayed(_parallel_build_trees)(
                t,
                model.bootstrap,
                X,
                y,
                sample_weight,
                i,
                len(trees),
                model.kwargs,
                model.train_classes,
                model.train_scores,
                model.feature_names_in_,
                verbose=model.verbose,
                class_weight=model.class_weight,
                n_samples_bootstrap=n_samples_bootstrap,
            )
            for i, t in enumerate(trees)
        )

        # Collect newly grown trees
        model.estimators_.extend(trees)

    if model.oob_score:
        y_type = type_of_target(y)
        if y_type in ("multiclass-multioutput", "unknown"):
            # FIXME: we could consider to support multiclass-multioutput if
            # we introduce or reuse a constructor parameter (e.g.
            # oob_score) allowing our user to pass a callable defining the
            # scoring strategy on OOB sample.
            raise ValueError(
                "The type of target cannot be used to compute OOB "
                f"estimates. Got {y_type} while only the following are "
                "supported: continuous, continuous-multioutput, binary, "
                "multiclass, multilabel-indicator."
            )
        model._set_oob_score_and_attributes(X, y)

    # Decapsulate classes_ attributes
    if hasattr(model, "classes_") and model.n_outputs_ == 1:
        model.n_classes_ = model.n_classes_[0]
        model.classes_ = model.classes_[0]
    return model


def biased_feature_importances_(model):
    """
    applying same function as the model's property, but setting all non-used features to importance=0
    """
    check_is_fitted(model)

    all_importances = Parallel(n_jobs=model.n_jobs, prefer="threads")(
        delayed(getattr)(tree, "biased_feature_importance")
        for tree in model.estimators_
        if tree.tree_.node_count > 1
    )

    if not all_importances:
        return np.zeros(model.n_features_in_, dtype=np.float64)

    all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
    return all_importances / np.sum(all_importances)


def apply_sauron(model, test_df):
    """
    Identify the trees to be used for regression prediction for each sample based on leaf classification assignment
    """

    samples_trees_distribution = {}
    for tree_idx, tree in enumerate(model.estimators_):
        # identify the leaf node for each sample
        samples_leaf_assignment = tree.apply(test_df.loc[:, tree.feature_names_in_])

        # fetch the class of each leaf node (hence, the class of the sample) as calculated during fitting
        samples_leaf_classes = [
            tree.leaf_class_assignments[leaf] for leaf in samples_leaf_assignment
        ]

        # store the index of the current tree to each sample as either sensitive or resistant tree,
        # based on leaf class assignment
        for sample_idx, class_assignment in enumerate(samples_leaf_classes):
            if class_assignment == 1:
                key = "sensitive_trees"
                other_key = "resistant_trees"
            elif class_assignment == 0:
                key = "resistant_trees"
                other_key = "sensitive_trees"
            else:
                raise TypeError("unexpected leaf assignment!")

            if sample_idx in samples_trees_distribution:
                samples_trees_distribution[sample_idx][key].append(tree_idx)
            else:
                samples_trees_distribution[sample_idx] = {
                    key: [tree_idx],
                    other_key: [],
                }

    # if a sample has majority sensitive trees, restrict the model regressors to these trees. Otherwise, use all trees
    samples_trees = {}
    rf_trees_indices = [idx for idx in range(len(model.estimators_))]
    for sample_idx, trees_dist in samples_trees_distribution.items():
        if len(trees_dist["sensitive_trees"]) >= len(trees_dist["resistant_trees"]):
            samples_trees[sample_idx] = trees_dist["sensitive_trees"]
        else:
            samples_trees[sample_idx] = rf_trees_indices
    return samples_trees


def predict_sample(X_df, sample_idx, trees_indices, model, y_hat):
    """
    called when sauron is used. The trees' indices are determined for each sample according to apply_sauron function
    """
    sample_estimates = 0
    for tree_idx in trees_indices:
        tree = model.estimators_[tree_idx]
        single_sample = (
            X_df.iloc[sample_idx][tree.feature_names_in_].to_frame().transpose()
        )
        sample_estimates += tree.predict(single_sample)
    y_hat[sample_idx] = sample_estimates / len(trees_indices)


def predict_sauron(model, X):
    check_is_fitted(model)

    # Check data
    X = model._validate_X_predict(X)

    # define the input dataframe with the features' names, to subset it properly for each tree
    X_df = pd.DataFrame(X, columns=model.feature_names_in_)

    # identify the trees to be used for each sample
    samples_trees = apply_sauron(model, X_df)

    # initialize y hat
    y_hat = np.zeros((X.shape[0]), dtype=np.float64)

    # predict each sample with respect to its corresponding trees, then average results
    for sample_idx, trees_indices in samples_trees.items():
        predict_sample(X_df, sample_idx, trees_indices, model, y_hat)
    return y_hat
