import json

import numpy as np
import pandas as pd
from manger.scoring_metrics.utils import get_sensitivity
from manger.utils import NewJsonEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    recall_score,
)


def calc_accuracy(
    true_labels: pd.Series,
    prediction: np.array,
    true_classes: pd.Series,
    regression: bool,
    output_file: str,
):
    """
    predefined accuracy scoring_metrics for classification and regression to be used for testing.
    """
    sens_true, sens_pred = get_sensitivity(true_classes, 1, true_labels, prediction)
    res_true, res_pred = get_sensitivity(true_classes, 0, true_labels, prediction)
    if output_file:
        raw_results = {
            "overall": {"true": true_labels, "pred": prediction},
            "sensitive": {"true": sens_true, "pred": sens_pred},
            "resistant": {"true": res_true, "pred": res_pred},
        }
        with open(output_file, "w") as raw:
            raw.write(json.dumps(raw_results, indent=2, cls=NewJsonEncoder))

    if regression:
        overall = mean_squared_error(true_labels, prediction)
        sensitive = mean_squared_error(sens_true, sens_pred)
        resistant = mean_squared_error(res_true, res_pred)
        acc = {"overall": overall, "sensitivity": sensitive, "specificity": resistant}
    else:
        recall = recall_score(true_labels, prediction, zero_division=1)
        f1 = f1_score(true_labels, prediction, zero_division=1)
        youden_j = balanced_accuracy_score(true_labels, prediction, adjusted=True)
        mcc = matthews_corrcoef(true_labels, prediction)

        # calculate overall specificity
        conf_mat = confusion_matrix(true_labels, prediction)
        tn, fp, fn, tp = conf_mat.ravel()
        specificity = tn / (tn + fp)
        acc = {
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "youden_j": youden_j,
            "mcc": mcc,
        }
    return acc
