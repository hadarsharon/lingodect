"""
Main module for direct interactive usage of the library
Wraps around the core library modules and serves the UI or CLI
"""
from argparse import ArgumentParser
from contextlib import closing

from detectors.text import MultinomialNBDetector
from loaders.text.massive import Massive
from loaders.text.clirmatrix import CLIRMatrix
from utils.db import SQLiteDB


def main():
    """
    Entrypoint for CLI
    """
    parser = ArgumentParser(prog="LingoDect")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="<string> opens and reads the specified text/image/audio file (relative/absolute path).")
    parser.add_argument("-s", "--string", type=str, default=None,
                        help="<string> reads the input string directly as text (wrap content with quotes).")
    parser.add_argument("-n", "--number", type=int, default=1,
                        help="<integer> number of languages to predict likelihood(s) for [default: 1]")
    parser.add_argument("-i", "--init", type=bool, default=False,
                        help="<boolean> whether or not to initialize the model and fit from scratch [default: False]")

    args = parser.parse_args()
    if bool(args.file) == bool(args.string):
        raise SystemExit("Bad arguments: expected exactly one of `file` (-f, --file) and `input` (-i, --input)")

    datasets = [Massive(), CLIRMatrix()]
    if args.init:
        detector = MultinomialNBDetector(datasets=datasets)
        detector.fit()
        detector.to_joblib()
    else:
        detector = MultinomialNBDetector.from_joblib(datasets=datasets)

    if args.number > 1:
        print(detector.predict_ranks(text=args.string, n=args.number))
    else:
        print(detector.predict(text=args.string))


if __name__ == '__main__':
    main()
