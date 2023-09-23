import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.naive_bayes import MultinomialNB

from detectors.text import MultinomialNBDetector
from loaders.text.clirmatrix import CLIRMatrix
from loaders.text.massive import Massive
from utils.config import Paths
from utils.datasets import BaseTextDataset

sns.set(rc={'figure.figsize': ("12", "19.2")})


@dataclass
class Evaluation:
    accuracy_score: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: dict


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


def plot_classification_report(
        results: Evaluation,
        figure_name: str = "classification_report.png",
        figures_dir: Optional[os.PathLike] = None
) -> None:
    df = pd.DataFrame(results.classification_report).iloc[:-1, :].T
    plt = sns.heatmap(df, annot=True)
    if figures_dir:
        figpath = str(Path(figures_dir) / figure_name)
        fig = plt.get_figure()
        fig.savefig(figpath)
        print(rf"Classification report figure saved to {figpath}")


def evaluate_multinomial_nb_on_datasets(datasets: list[BaseTextDataset]) -> dict[BaseTextDataset, Evaluation]:
    print(rf"Starting evaluations on Multinomial Naive Bayes classifier model using dataset(s): {datasets}")
    print('⸻' * 25)
    multinomial_nb = MultinomialNBDetector.from_joblib(datasets=datasets)
    model: MultinomialNB = multinomial_nb.model
    evaluations: dict[BaseTextDataset, Evaluation] = {}
    for dataset in datasets:
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
        matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print("Building classification report...")
        report = classification_report(y_true=y_test, y_pred=y_pred, labels=list(set(y_test)), output_dict=True)
        labels = []
        for cls in report.keys():
            if cls.isnumeric():
                # transform numeric classes (e.g. 128) to language code (e.g. 'af')
                labels.append((cls, multinomial_nb.label_encoder.inverse_transform([int(cls)])[0]))
        for cls, transformed in labels:
            report[transformed] = report.pop(cls)
        print("Evaluation results are ready!")
        print('⸻' * 25)
        evaluations[dataset] = Evaluation(
            accuracy_score=accuracy,
            f1_score=f1,
            confusion_matrix=matrix,
            classification_report=report
        )
    return evaluations


if __name__ == "__main__":
    massive = Massive()
    clirmatrix = CLIRMatrix()
    evaluations = evaluate_multinomial_nb_on_datasets(datasets=[massive, clirmatrix])
    i = 0
    for dataset, evaluation in evaluations.items():
        plot_classification_report(
            results=evaluation,
            figures_dir=Paths.EVALUATIONS / "figures", figure_name=rf"classification_report_{i}.png"
        )
        i += 1
