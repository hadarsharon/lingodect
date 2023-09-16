import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier

from config import Paths


class ECAPA_TDNN:  # noinspection PyPep8Naming
    MODEL_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
    MODEL_DIRECTORY = Paths.MODELS / "speech/ecapa_tdnn"

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe_speech(self, audio_path: str, language: str) -> str:
        with sr.AudioFile(audio_path) as source:
            audio_data = self.recognizer.listen(source)
            return self.recognizer.recognize_google(audio_data=audio_data, language=language)

    @staticmethod
    def detect_speech_language(audio_path: str, save_dir: str = MODEL_DIRECTORY) -> list[str]:
        language_id = EncoderClassifier.from_hparams(
            source=ECAPA_TDNN.MODEL_SOURCE,
            savedir=save_dir
        )
        signal = language_id.load_audio(audio_path)
        prediction = language_id.classify_batch(signal)
        return prediction[3]
