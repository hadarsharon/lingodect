import webbrowser
from collections import namedtuple
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

from detectors.speech import ECAPA_TDNN
from detectors.text import MultinomialNBDetector
from loaders.text.clirmatrix import CLIRMatrix
from loaders.text.massive import Massive
from utils.config import Paths

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

app_params_fields = ["detection_text", "language_codes", "language_names", "transcript", "content_type", "exception"]
app_params_defaults = {
    "detection_text": None,
    "language_codes": [],
    "language_names": [],
    "transcript": False,
    "content_type": "unknown file",
    "errors": None
}

AppParams = namedtuple(
    typename="AppParams",
    field_names=app_params_fields,
    defaults=[app_params_defaults.get(field) for field in app_params_fields]
)

fake = Faker(FAKE_TEXT_LOCALES)
fake_texts = [fake.text() for _ in range(FAKE_TEXTS)]
datasets = [Massive(), CLIRMatrix()]
text_detector = MultinomialNBDetector.from_joblib(datasets=datasets)
speech_detector = ECAPA_TDNN()


def open_browser():
    webbrowser.open_new(rf"http://{APP_HOST}:{APP_PORT}/")


def get_language_name(language_code: str) -> str:
    language = pycountry.languages.get(alpha_2=language_code) or pycountry.languages.get(alpha_3=language_code)
    return getattr(language, "name", "Unknown")


def process_text(
        text_input: Optional[str] = None,
        file_input: Optional[FileStorage] = None,
        multi_language: bool = False
):
    assert bool(text_input) != bool(file_input), "Exactly one of `text_input`, `file_input` must be supplied!"

    if file_input:
        text_input: str = file_input.stream.read().decode("utf-8")

    if multi_language:
        language_codes: list[str] | list[tuple[int, str]] = text_detector.predict_probabilities(text=text_input)
        if len(language_codes) == 1:  # singleton
            language_names: list[str] = [get_language_name(language_code=language_codes[0])]
        else:  # multiple
            language_names: list[str] = [get_language_name(language_code=t[1]) for t in language_codes]
    else:
        language_codes: list[str] = text_detector.predict(text=text_input)
        language_names: list[str] = [get_language_name(language_code=language_codes[0])]

    return AppParams(detection_text=text_input, language_codes=language_codes, language_names=language_names,
                     content_type="text file" if file_input else "text")  # noqa


def process_audio(file_input: FileStorage, transcribe: bool = False) -> AppParams:
    with NamedTemporaryFile(dir=Paths.DATASETS / "speech", suffix=Path(file_input.filename).suffix, delete=True) as tf:
        tf.write(file_input.stream.read())
        predictions: list[str] = speech_detector.detect_speech_language(audio_path=tf.name) or []
        if predictions:
            language_codes, language_names = ([str.strip(pred)] for pred in predictions[0].split(r':'))
            transcript = speech_detector.transcribe_speech(
                audio_path=tf.name,
                language=language_codes[0]
            ) if transcribe else None
        else:
            language_codes, language_names, transcript = ["?"], ["Unknown"], None

    return AppParams(
        detection_text=file_input.filename,
        language_codes=language_codes,
        language_names=language_names,
        transcript=transcript,
        content_type="audio file",
        exception=None
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
        raise NotImplementedError(f"Unknown MIME-Type received: {file_input.mimetype}")


def process_input(req: request):
    try:
        text_input: str = req.form.get("textInput")
        file_input: FileStorage = req.files.get("fileInput")
        params: AppParams
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

    except Exception as e:
        return render_template(
            "index.html",
            exception=e,
            zip=zip
        )

    else:
        return render_template(
            "index.html",
            fake_texts=fake_texts,
            country_codes=country_codes,
            detection_text=params.detection_text,
            language_codes=params.language_codes,
            language_names=params.language_names,
            content_type=params.content_type,
            transcript=params.transcript,
            exception=params.exception,
            zip=zip
        )


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", fake_texts=fake_texts, zip=zip)
    else:
        return process_input(req=request)


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(host=APP_HOST, port=APP_PORT)
