<h1 align="center">
FastVC
</h1>

## Overview

FastVC is a fast and efficient, *non-parallel* and *any-to-any* *voice conversion (VC)* tool. Our tool exploits the task by cascading an *Automatic Speech Recognition (ASR)* model and a *Text To Speech (TTS)* model.

<p align="center">
  <img src="https://user-images.githubusercontent.com/17434626/122647192-44026400-d123-11eb-9b56-305b312744d7.png" width="85%"//>
</p>

The ASR is based on [Wav2vec 2.0](https://arxiv.org/pdf/2006.11477.pdf) and is used to transcribe the speech from a *source speaker*. The TTS is based on [SV2TTS](https://arxiv.org/pdf/1806.04558.pdf) and is used to generate the output speech from a *target speaker* embedding.

For a more detailed explanation checkout [the paper of our project]().

## Installation & usage

The software was implemented using `python 3.9.4`

1. (*optional*) Create virtual env and activate it: `python -m venv env` and `source env/bin/activate` (if using macOS/Linux) or `.\env\Scripts\activate` (if using Windows)
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Download the [pretrained models](https://github.com/blue-fish/Real-Time-Voice-Cloning/releases/download/v1.0/pretrained.zip) and put them in the correct directories:
  ````
  src/encoder/saved_models/pretrained.pt
  src/synthesizer/saved_models/pretrained/pretrained.pt
  src/vocoder/saved_models/pretrained/pretrained.pt
  ````
5. Run the main script: `python src/main.py` (use `--help` for displaying available options)

More instructions can be found [here](https://github.com/fmiotello/fastVC/tree/main/src).

## Notes

This application was developed as a project for the "Project Course" at [Politecnico di Milano](https://www.polimi.it/en/) (MSc in Music and Acoustic Engineering).

*[Luigi Attorresi](https://github.com/LuigiAttorresi)*<br>
*[Federico Miotello](https://github.com/fmiotello)*<br>
*[Eugenio Poliuti](https://github.com/Poliuti)*<br>
