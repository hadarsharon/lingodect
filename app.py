import webbrowser
from collections import namedtuple
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Timer
from typing import Iterable, Optional

import pandas as pd
import pycountry
from faker import Faker
from faker.config import AVAILABLE_LOCALES
from flask import Flask, request, render_template
from werkzeug.datastructures import FileStorage

from config import Paths
from detectors.speech import ECAPA_TDNN
from detectors.text import MultinomialNBDetector
from utils.datasets import Massive, CLIRMatrix
from utils.db import SQLiteDB

LOCALES_DATAFRAME = pd.DataFrame([locale.split(r'_') for locale in AVAILABLE_LOCALES],
                                 columns=["language_code", "country_code"]).drop_duplicates(ignore_index=True)
FAKE_TEXT_LOCALES = (
    "ar_AA",  # Arabic
    "az_AZ",  # Azerbaijani
    "bn_BD",  # Bengali
    "cs_CZ",  # Czech
    "da_DK",  # Danish
    "de_AT",  # German (Austria)
    "de_DE",  # German (Germany)
    "el_GR",  # Greek
    "en_PH",  # English (Philippines)
    "en_US",  # English (USA)
    "fa_IR",  # Farsi
    "fil_PH",  # Filipino
    "fr_FR",  # French
    "he_IL",  # Hebrew
    "hy_AM",  # Armenian
    "ja_JP",  # Japanese
    "la",  # Latin
    "nl_BE",  # Dutch (Belgium)
    "nl_NL",  # Dutch (Netherlands)
    "pl_PL",  # Polish
    "ru_RU",  # Russian
    "th_TH",  # Thai
    "tl_PH",  # Tagalog
    "zh_CN",  # Chinese (Simplified)
    "zh_TW"  # Chinese (Traditional)
)

FAKE_TEXTS = 1000

APP_HOST: str = "localhost"
APP_PORT: int = 8080

app = Flask(__name__)

app_params_fields = ["detection_text", "language_codes", "language_names", "transcript", "content_type"]
app_params_defaults = {
    "detection_text": None,
    "language_codes": [],
    "language_names": [],
    "transcript": False,
    "content_type": "unknown file"
}

AppParams = namedtuple(
    typename="AppParams",
    field_names=app_params_fields,
    defaults=[app_params_defaults.get(field) for field in app_params_fields]
)


def open_browser():
    webbrowser.open_new(rf"http://{APP_HOST}:{APP_PORT}/")


def process_text(
        text_input: Optional[str] = None,
        file_input: Optional[FileStorage] = None,
        multi_language: bool = False
):
    assert bool(text_input) != bool(file_input), "Exactly one of `text_input`, `file_input` must be supplied!"

    if multi_language:
        language_codes: list[str] | list[tuple[int, str]] = text_detector.predict_probabilities(text=text_input)
        if len(language_codes) == 1:  # singleton
            language_names: list[str] = [getattr(pycountry.languages.get(alpha_2=language_codes[0]), "name", "Unknown")]
        else:  # multiple
            language_names: list[str] = [
                getattr(pycountry.languages.get(alpha_2=t[1]), "name", "Unknown") for t in language_codes
            ]
    else:
        language_codes: list[str] = text_detector.predict(text=text_input)
        language_names: list[str] = [getattr(pycountry.languages.get(alpha_2=language_codes[0]), "name", "Unknown")]

    return AppParams(
        detection_text=text_input,
        language_codes=language_codes,
        language_names=language_names,
        content_type="text file" if file_input else "text"
    )


def process_audio(file_input: FileStorage, transcribe: bool = False) -> AppParams:
    with NamedTemporaryFile(dir=Paths.DATASETS / "speech", suffix=Path(file_input.name).suffix, delete=True) as tf:
        tf.write(file_input.stream.read())
        predictions: list[str] = speech_detector.detect_speech_language(audio_path=tf.name) or []
        if predictions:
            language_codes, language_names = ([str.strip(pred)] for pred in predictions[0].split(r':'))
            if transcribe:
                transcript = speech_detector.transcribe_speech(audio_path=tf.name, language=language_codes[0])
        else:
            language_codes, language_names, transcript = ["?"], ["Unknown"], None

    return AppParams(
        detection_text=file_input.name,
        language_codes=language_codes,
        language_names=language_names,
        transcript=transcript,
        content_type="audio file"
    )


def process_image():
    ...


def process_file(req: request, file_input: FileStorage) -> AppParams:
    if "text" in file_input.mimetype:
        return process_text(
            file_input=file_input,
            multi_language=req.form.get("multiLanguageSwitch", "").lower() == "on"
        )
    elif "image" in file_input.mimetype:
        return process_image()
    elif "audio" in file_input.mimetype:
        return process_audio(
            file_input=file_input,
            transcribe=request.form.get("speechTranscriptionSwitch", "").lower() == "on"
        )
    else:
        return  # TODO: do something


def process_input(req: request):
    text_input: str = req.form.get("textInput")
    file_input: FileStorage = req.files.get("fileInput")
    if file_input:
        params = process_file(req=req, file_input=file_input)
    elif text_input:
        params = process_text(
            text_input=text_input,
            multi_language=req.form.get("multiLanguageSwitch", "").lower() == "on"
        )
    else:
        ...  # TODO: error handling

    country_codes: Iterable[str] = filter(
        None,
        LOCALES_DATAFRAME[
            LOCALES_DATAFRAME["language_code"].isin(params.language_codes)].country_code.unique().tolist()
    )

    return render_template(
        "index.html",
        fake_texts=fake_texts,
        country_codes=country_codes,
        detection_text=params.detection_text,
        language_codes=params.language_codes,
        language_names=params.language_names,
        content_type=params.content_type,
        transcript=params.transcript,
        zip=zip
    )


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", fake_texts=fake_texts, zip=zip)
    else:  # TODO: functinos
        if file_input:
            ...
        elif text_input:
            ...
        else:
            ...  # TODO: error handling
            return render_template("index.html")


if __name__ == "__main__":
    db = SQLiteDB()
    fake = Faker(FAKE_TEXT_LOCALES)
    fake_texts = [fake.text() for _ in range(FAKE_TEXTS)]
    with closing(db.get_connection()) as conn:
        datasets = [Massive(sqlite_conn=conn), CLIRMatrix(sqlite_conn=conn)]
        text_detector = MultinomialNBDetector.from_pickle(datasets=datasets)
    speech_detector = ECAPA_TDNN()
    Timer(1, open_browser).start()
    app.run(host=APP_HOST, port=APP_PORT, debug=True)
