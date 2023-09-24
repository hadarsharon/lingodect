# LingoDect

A Language Detection (LID) library written in Python, able to detect language from text, image or audio content.

## Tech Stack

**Back-end**: Python (numpy, pandas, etc.)

**Server:** Flask

**Models:** scikit-learn (sklearn), TensorFlow, Keras

**Front-end:** HTML, CSS (Bootstrap), vanilla JavaScript

## Installation

Install LingoDect with pip

Clone this repository, and from the repository root run the following command:

```bash
  pip install -e .
```

You should now have lingodect installed as a library and can use it for development and testing.

## Run Locally

Clone the project

```bash
  git clone https://github.com/hadarsharon/lingodect
```

Go to the project directory

```bash
  cd lingodect
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```

Alternatively, you can run commands via the CLI
(run with the -h flag for help and information about available commands)

```bash
  python cli.py -h
```

## Running Tests

To run tests, run the following command from the project root

```bash
  pytest
```

## FAQ

#### What types of inputs does this library support?

The library currently supports all textual input that can be written directly to it (whether as a string via the CLI or
in a text box via the web application GUI), or via plaintext files such as `.txt` files.

Audio (Speech) and Image (Handwriting) inputs are available using most common input
formats (`.wav`, `.flac`, `.jpg`, `.png` etc.).

In case your input file format is not supported, `.wav` and `.png` are a safe bet, so you should convert your audio or
image file to them, respectively.

#### How many languages does this library support for prediction purposes?

The library models are currently trained on over 100 languages, so there is a good chance whatever language you want to
predict is part of the support languages.
