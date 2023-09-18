import logging
import sys
from functools import cached_property
from typing import Literal

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import EagerTensor

from utils.config import Paths
from utils.datasets import BaseImageDataset

logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)

AUTOTUNE = tf.data.AUTOTUNE


class IAM(BaseImageDataset):
    IMAGES_DIR = Paths.DATASETS / "image/IAM/words"
    METADATA_FILE = Paths.DATASETS / "image/IAM/words.txt"
    IMAGE_TRANSCRIPTION_KEY = "transcription"
    IMAGE_PATH_KEY = "img_path"

    def __init__(self, train_size: float = 0.9, test_dev_size: float = 0.1, batch_size: int = 64):
        self.train, self.test, self.dev = (
            self.get_train_test_dev_dataset(train_size=train_size, test_dev_size=test_dev_size)
        )
        self.batch_size = batch_size
        super().__init__(image_transcription_key=self.IMAGE_TRANSCRIPTION_KEY)

    @staticmethod
    def read_dataframe(
            filter_segmentation_errors: bool = False,
            filter_empty_transcriptions: bool = False,
            shuffle_data: bool = False,
            parse_img_paths: bool = True
    ) -> pd.DataFrame:
        df = pd.read_table(
            IAM.METADATA_FILE,
            delim_whitespace=True,
            comment="#",
            on_bad_lines="skip",
            names=[
                "word_id",
                "segmentation",
                "graylevel",
                *(rf"bounding_box_{_}" for _ in ("x", "y", "w", "h")),
                "grammatical_tag",
                "transcription"
            ],
            quotechar='^'
        )
        if filter_segmentation_errors:
            df = df[df["segmentation"] == "ok"]
        if filter_empty_transcriptions:
            df = df.dropna(axis=0, how="any", subset="transcription")
        if shuffle_data:
            df = df.sample(frac=1).reset_index(drop=True)
        if parse_img_paths:
            path_blocks = df["word_id"].str.split(r'-')
            df["img_path"] = [
                rf"{IAM.IMAGES_DIR}/{blocks[0]}/{r'-'.join(blocks[:2])}/{r'-'.join(blocks)}.png"
                for blocks in path_blocks
            ]
        return df

    def get_train_test_dev_dataset(
            self,
            train_size: float = 0.9,
            test_dev_size: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.read_dataframe(
            filter_segmentation_errors=True,
            filter_empty_transcriptions=True,
            shuffle_data=True,
            parse_img_paths=True
        )
        train, test = train_test_split(df, train_size=train_size, test_size=test_dev_size)
        test, dev = train_test_split(test, train_size=0.5, test_size=0.5)
        return train, test, dev

    def get_image_paths(self, partition: Literal["train", "test", "dev"]) -> list[str]:
        return getattr(self, partition)[self.image_path_key].tolist()

    def get_labels(self, partition: Literal["train", "test", "dev"]) -> list[str]:
        return getattr(self, partition)[self.image_transcription_key].tolist()

    @cached_property
    def max_label_length_train(self) -> int:
        return max(map(len, self.get_labels(partition="train")))

    def get_vocabulary(self, partition: Literal["train", "test", "dev"]) -> set[str]:
        return {  # TODO: lowercase? uppercase?
            character for transcription in getattr(self, partition)[self.image_transcription_key].str.split(" ")
            for word in transcription for character in word
        }

    def vocabulary_size(self, partition: Literal["train", "test", "dev"]) -> int:
        return len(self.get_vocabulary(partition=partition))

    def preprocess_image(self, image_path) -> EagerTensor:
        png = tf.io.read_file(image_path)
        decoded = tf.image.decode_png(contents=png, channels=1)
        image = tf.cast(decoded, tf.float32) / 255.0
        return image

    def vectorize_label(self, label: str) -> EagerTensor:
        padding_token = 99
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return tf.pad(
            tensor=label,
            paddings=[[0, self.max_label_length_train - len(label)]],
            constant_values=padding_token
        )

    def process_images_labels(self, image_path, label) -> dict[str, EagerTensor]:
        return {"image": self.preprocess_image(image_path=image_path), "label": self.vectorize_label(label=label)}

    def prepare_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            tensors=(self.get_image_paths(partition="train"), self.get_labels(partition="train"))
        ).map(
            self.process_images_labels, num_parallel_calls=AUTOTUNE
        )
        return dataset.batch(batch_size=self.batch_size).cache().prefetch(AUTOTUNE)

    @cached_property
    def train_vocabulary(self) -> set[str]:
        return self.get_vocabulary(partition="train")

    @cached_property
    def test_vocabulary(self) -> set[str]:
        return self.get_vocabulary(partition="test")

    @cached_property
    def dev_vocabulary(self) -> set[str]:
        return self.get_vocabulary(partition="dev")

    @cached_property
    def train_labels(self):
        return self.get_labels(partition="train")

    @cached_property
    def test_labels(self):
        return self.get_labels(partition="test")

    @cached_property
    def dev_labels(self):
        return self.get_labels(partition="dev")
