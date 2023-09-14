import webbrowser
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Timer
from typing import Iterable

import pandas as pd
import pycountry
from faker import Faker
from faker.config import AVAILABLE_LOCALES
from flask import Flask, request, render_template
from werkzeug.datastructures import FileStorage

from config import Paths
from detectors.speech import detect_speech_language, transcribe_speech
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


def detect(text: str) -> list[str]:
    return detector.predict(text=text)


def open_browser():
    webbrowser.open_new(rf"http://{APP_HOST}:{APP_PORT}/")


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", fake_texts=fake_texts, zip=zip)
    else:  # TODO: functinos
        text_input: str = request.form.get("textInput")
        file_input: FileStorage = request.files.get("fileInput")
        transcript = None  # overridden later if relevant
        if file_input:
            # TODO: mimetype, contenttype etc. to differentiate between image and audio
            # Assuming we have audio
            content_type = "audio file"
            transcribe = request.form.get("speechTranscriptionSwitch", "").lower() == "on"
            text_input = file_input.filename
            suffix = Path(file_input.filename).suffix
            with NamedTemporaryFile(dir=Paths.DATASETS / "speech", suffix=suffix, delete=True) as tf:
                tf.write(file_input.stream.read())
                predictions: list[str] = detect_speech_language(audio_path=tf.name) or []
                if predictions:
                    language_codes, language_names = ([str.strip(pred)] for pred in predictions[0].split(r':'))
                    if transcribe:
                        transcript = transcribe_speech(audio_path=tf.name, language=language_codes[0])
                else:
                    language_codes, language_names, transcript = ["?"], ["Unknown"], None
        elif text_input:
            content_type = "text"
            match request.form.get("multiLanguageSwitch", "").lower():  # TODO: functinos
                case "on":  # multi language
                    language_codes: list[str] | list[tuple[int, str]] = detector.predict_probabilities(text=text_input)
                    if len(language_codes) == 1:  # singleton
                        language_names: list[str] = [
                            getattr(pycountry.languages.get(alpha_2=language_codes[0]), "name", "Unknown")
                        ]
                    else:  # multiple
                        language_names: list[str] = [
                            getattr(pycountry.languages.get(alpha_2=t[1]), "name", "Unknown")
                            for t in language_codes
                        ]
                case _:  # single language
                    language_codes: list[str] = detect(text=text_input)
                    language_names: list[str] = [
                        getattr(pycountry.languages.get(alpha_2=language_codes[0]), "name", "Unknown")]
        else:
            ...  # TODO: error handling
            return render_template("index.html")

        country_codes: Iterable[str] = filter(
            None,
            LOCALES_DATAFRAME[
                LOCALES_DATAFRAME["language_code"].isin(language_codes)].country_code.unique().tolist()
        )

        return render_template(
            "index.html",
            fake_texts=fake_texts,
            detection_text=text_input,
            language_codes=language_codes,
            language_names=language_names,
            country_codes=country_codes,
            content_type=content_type,
            transcript=transcript,
            zip=zip
        )


if __name__ == "__main__":
    db = SQLiteDB()
    fake = Faker(FAKE_TEXT_LOCALES)
    fake_texts = [fake.text() for _ in range(FAKE_TEXTS)]
    with closing(db.get_connection()) as conn:
        datasets = [Massive(sqlite_conn=conn), CLIRMatrix(sqlite_conn=conn)]
        detector = MultinomialNBDetector.from_pickle(datasets=datasets)
    Timer(1, open_browser).start()
    app.run(host=APP_HOST, port=APP_PORT)
