"""
Module for defining textual language detection models and their full implementation(s).
These models are used with Datasets from the datasets.py module, to perform language classification on text.
"""

import logging
import sys
from typing import Optional

import dill
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from config import Paths
from utils.datasets import BaseTextDataset

logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)


class MultinomialNBDetector:
    """
    Detection model based on the popular Multinomial Naive Bayes algorithm.
    """
    MODEL_PICKLE = Paths.MODELS / "text/multinomial_nb"

    def __init__(
            self,
            datasets: BaseTextDataset | list[BaseTextDataset],
            minibatches: Optional[int] = None
    ):
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.minibatches = minibatches or 1
        self.hashing_vectorizer = HashingVectorizer(decode_error="ignore", n_features=(2 ** 22), alternate_sign=False)
        self.label_encoder = LabelEncoder()
        self.model = MultinomialNB()

    def bag_of_words(
            self,
            df: pd.DataFrame,
            text_key: str
    ) -> np.ndarray[np.ndarray[np.int64]]:
        x = df[text_key]
        return self.hashing_vectorizer.fit_transform(x).toarray()

    def iter_minibatches(self):
        """
        Generator of minibatches for operating on larger-than-memory datasets.
        """
        for dataset in self.datasets:
            minibatches = np.array_split(dataset.train_dataset, self.minibatches)
            for df_ in minibatches:
                X_text, y = df_[dataset.TEXT_KEY], self.label_encoder.transform(df_[dataset.LANGUAGE_KEY])
                yield X_text, y

    def fit(self):
        logger.info("Begin fitting model.")
        all_classes = np.unique(
            self.label_encoder.fit_transform(
                np.concatenate([dataset.train_dataset[dataset.LANGUAGE_KEY] for dataset in self.datasets])
            )
        )
        minibatches_iterator = self.iter_minibatches()
        for i, (X_train_text, y_train) in enumerate(minibatches_iterator, start=1):
            logger.info(rf"Fit iteration {i}...")
            X_train = self.hashing_vectorizer.transform(X_train_text)
            if self.minibatches > 1 or len(self.datasets) > 1:
                self.model.partial_fit(X_train, y_train, classes=all_classes)
            else:
                self.model.fit(X_train, y_train)

        logger.info("Done fitting model.")

        for i, dataset in enumerate(self.datasets):
            test_df = dataset.test_dataset
            X_test_text, y_test = test_df[dataset.TEXT_KEY], self.label_encoder.transform(test_df[dataset.LANGUAGE_KEY])
            X_test = self.hashing_vectorizer.transform(X_test_text)
            logger.info(rf"Accuracy score (test/dataset {i}): {self.model.score(X_test, y_test)}")

    def predict(self, text: str) -> str:
        X = self.hashing_vectorizer.transform([text]).toarray()
        y = self.model.predict(X)
        return self.label_encoder.inverse_transform(y)

    @staticmethod
    def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
        return (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))

    def predict_probabilities(self, text: str, threshold: float = 0.2) -> list[str] | list[tuple[int, str]]:
        X = self.hashing_vectorizer.transform([text]).toarray()
        y = self.model.predict_proba(X)
        normalized = self._normalize_probabilities(probabilities=y)
        probabilities = sorted(list(filter(lambda p: p[0] > threshold, zip(normalized[0], self.model.classes_))),
                               key=lambda t: t[0], reverse=True)
        if len(probabilities) == 0:  # none passed threshold?
            return ["?"]
        elif len(probabilities) == 1:  # one certain language
            return list(self.label_encoder.inverse_transform([probabilities[0][1]]))
        else:  # multiple probabilities
            return [(int(t[0] * 100), self.label_encoder.inverse_transform([t[1]])[0]) for t in probabilities]

    def predict_ranks(self, text: str, n: int = 3) -> list[tuple[int, str]]:
        X = self.hashing_vectorizer.transform([text]).toarray()
        y = self.model.predict_proba(X)
        probabilities = sorted(list(zip(y[0], self.model.classes_)), key=lambda t: t[0], reverse=True)[:n]
        return [(i, self.label_encoder.inverse_transform([t[1]])[0]) for i, t in enumerate(probabilities, start=1)]

    @classmethod
    def from_pickle(
            cls,
            datasets: BaseTextDataset | list[BaseTextDataset],
            minibatches: Optional[int] = None
    ):
        klass = cls(datasets=datasets, minibatches=minibatches)
        with open(cls.MODEL_PICKLE / "model.pickle", "rb") as f:
            klass.model = dill.load(f)
        with open(cls.MODEL_PICKLE / "label_encoder.pickle", "rb") as f:
            klass.label_encoder = dill.load(f)
        with open(cls.MODEL_PICKLE / "hashing_vectorizer.pickle", "rb") as f:
            klass.hashing_vectorizer = dill.load(f)
        return klass

    def to_pickle(self):
        with open(self.MODEL_PICKLE / "model.pickle", "wb") as f:
            dill.dump(self.model, f)
        with open(self.MODEL_PICKLE / "label_encoder.pickle", "wb") as f:
            dill.dump(self.label_encoder, f)
        with open(self.MODEL_PICKLE / "hashing_vectorizer.pickle", "wb") as f:
            dill.dump(self.hashing_vectorizer, f)
