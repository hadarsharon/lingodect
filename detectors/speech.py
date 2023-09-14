import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier

from config import Paths


def transcribe_speech(audio_path: str, language: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.listen(source)
        return recognizer.recognize_google(audio_data=audio_data, language=language)


def detect_speech_language(audio_path: str) -> list[str]:
    language_id = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=str(Paths.MODELS / "speech/speechbrain")
    )
    signal = language_id.load_audio(audio_path)
    prediction = language_id.classify_batch(signal)
    return prediction[3]
