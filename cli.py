"""
Main module for direct interactive usage of the library
Wraps around the core library modules and serves the UI or CLI
"""
import mimetypes
from argparse import ArgumentParser
from werkzeug.datastructures import FileStorage


def main():
    """
    Entrypoint for CLI
    """
    parser = ArgumentParser(prog="LingoDect")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="<string> opens and reads the specified text/image/audio file (absolute path).")
    parser.add_argument("-s", "--string", type=str, default=None,
                        help="<string> reads the input string directly as text (wrap content with quotes).")
    parser.add_argument("-m", "--multi", type=bool, default=False,
                        help="<boolean> whether or not to predict likelihood(s) for multiple possible languages "
                             "(ignored for non-text input) [default: False]")
    parser.add_argument("-t", "--transcribe", type=bool, default=False,
                        help="<boolean> whether or not to transcribe speech in audio content to text "
                             "(ignored for non-audio input) [default: False]")
    parser.add_argument("-i", "--init", type=bool, default=False,
                        help="<boolean> whether or not to initialize the Multinomial NB model "
                             "and fit from scratch [default: False]")

    args = parser.parse_args()
    if bool(args.file) == bool(args.string):
        raise SystemExit("Bad arguments: expected exactly one of `file` (-f, --file) and `input` (-i, --input)")

    if args.init:
        from loaders.text.clirmatrix import CLIRMatrix
        from loaders.text.massive import Massive
        from detectors.text import MultinomialNBDetector
        datasets = [Massive(), CLIRMatrix()]
        detector = MultinomialNBDetector(datasets=datasets)
        detector.fit()
        detector.to_joblib()

    print('⸻' * 25)
    print("Running language detection ...")
    print('*' * 25)
    if args.string:
        from app import process_text
        prediction = process_text(text_input=args.string, multi_language=args.multi)
        print('⸻' * 25)
        print(rf"Detected language(s): {prediction.language_names} ({prediction.language_codes})")
    else:  # file
        mimetype = mimetypes.guess_type(args.file)[0]
        if "text" in mimetype:
            from app import process_text
            with open(args.file, 'rb') as f:
                prediction = process_text(file_input=FileStorage(f), multi_language=args.multi)
                print('⸻' * 25)
                print(rf"Detected language(s): {prediction.language_names} ({prediction.language_codes})")
        elif "audio" in mimetype:
            from app import process_audio
            with open(args.file, 'rb') as f:
                prediction = process_audio(file_input=FileStorage(f), transcribe=args.transcribe)
                print('⸻' * 25)
                print(rf"Detected language(s): {prediction.language_names} ({prediction.language_codes})")
                if args.transcribe:
                    print('⸻' * 25)
                    print(f'Speech-to-Text transcription: "{prediction.transcript}"')
        else:
            raise NotImplementedError(
                rf"Unknown content/mime type {mimetype} in file: {args.file} - "
                "perhaps file is invalid or format is not supported."
            )
    print()


if __name__ == '__main__':
    main()
