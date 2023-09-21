import glob
from pathlib import Path

import pytest
from werkzeug.datastructures import FileStorage

from app import get_language_name, process_text, process_audio
from utils.config import Paths


def test_get_language_name_iso2_valid():
    assert get_language_name(language_code="he") == "Hebrew"


def test_get_language_name_iso3_valid():
    assert get_language_name(language_code="arz") == "Egyptian Arabic"


def test_get_language_name_unknown():
    assert get_language_name(language_code="zz") == "Unknown"


def test_process_text_input():
    prediction = process_text(text_input="This is a test for the application", multi_language=False)
    assert prediction and prediction.language_codes and prediction.language_codes[0] == "en"


@pytest.mark.parametrize(
    ("file_input", "expected"),
    [(file_path, language_code) for file_path, language_code in zip(
        glob.glob(str(Paths.TESTS / "test_inputs/text/*.txt")),
        [Path(file).stem for file in glob.glob(str(Paths.TESTS / "test_inputs/text/*.txt"))]
    )]
)
def test_process_text_file(file_input, expected):
    with open(file_input, 'rb') as f:
        prediction = process_text(file_input=FileStorage(f), multi_language=False)
        assert prediction and prediction.language_codes and prediction.language_codes[0] == expected


@pytest.mark.parametrize(
    ("file_input", "expected"),
    [(file_path, language_code) for file_path, language_code in zip(
        glob.glob(str(Paths.TESTS / "test_inputs/speech/*.wav")),
        [Path(file).stem for file in glob.glob(str(Paths.TESTS / "test_inputs/speech/*.wav"))]
    )]
)
def test_process_audio_file(file_input, expected):
    with open(file_input, 'rb') as f:
        prediction = process_audio(file_input=FileStorage(f))
        assert prediction and prediction.language_codes and prediction.language_codes[0] == expected
