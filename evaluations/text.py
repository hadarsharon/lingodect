import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score, ConfusionMatrixDisplay
)
from sklearn.naive_bayes import MultinomialNB

from detectors.text import MultinomialNBDetector
from loaders.text.clirmatrix import CLIRMatrix
from loaders.text.massive import Massive
from utils.datasets import BaseTextDataset


@dataclass
class Evaluation:
    accuracy_score: float
    f1_score: float
    confusion_matrix: np.ndarray


def plot_confusion_matrix(
        results: Evaluation,
        figure_name: str = "confusion_matrix.png",
        figures_dir: Optional[os.PathLike] = None
) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=results.confusion_matrix)
    plt = disp.plot()
    if figures_dir:
        figpath = str(Path(figures_dir) / figure_name)
        plt.figure_.savefig(figpath)
        print(rf"Confusion matrix figure saved to {figpath}")


def evaluate_multinomial_nb_on_dataset(dataset: BaseTextDataset) -> Evaluation:
    print(rf"Starting evaluations on Multinomial Naive Bayes classifier model using dataset: {type(dataset)}")
    print('⸻' * 25)
    multinomial_nb = MultinomialNBDetector.from_joblib(datasets=[dataset])
    model: MultinomialNB = multinomial_nb.model
    print(rf"Getting test dataset from {type(dataset)} ...")
    test_ds: pd.DataFrame = dataset.test_dataset
    print(rf"Obtaining ground truths (X_test, y_test) ...")
    X_test: csr_matrix = multinomial_nb.hashing_vectorizer.transform(test_ds[dataset.TEXT_KEY])
    y_test: np.ndarray = multinomial_nb.label_encoder.transform(test_ds[dataset.LANGUAGE_KEY])
    print("Predicting y_pred ...")
    y_pred: np.ndarray = model.predict(X_test)
    print("Calculating accuracy score ...")
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Calculating f1 score ...")
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
    print("Calculating confusion matrix ...")
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=list(set(y_test)))
    print("Evaluation results are ready!")
    print('⸻' * 25)
    return Evaluation(
        accuracy_score=accuracy,
        f1_score=f1,
        confusion_matrix=matrix
    )


if __name__ == "__main__":
    massive = Massive()
    clirmatrix = CLIRMatrix()
    evaluation = evaluate_multinomial_nb_on_dataset(dataset=massive)
