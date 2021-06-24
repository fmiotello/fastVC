<h1 align="center">
FastVC
</h1>

## Overview

FastVC is a fast and efficient, non-parallel and any-to-any *voice conversion (VC)* tool. VC involves the modification of the voice of a source speaker to make it sound like that of a target speaker, without changing the linguistic content of the sentence. Our tool exploits the task by cascading an Automatic Speech Recognition (ASR) model and a Text To Speech (TTS) model.

<p align="center">
  <img src="https://user-images.githubusercontent.com/17434626/122674111-014c9480-d1d4-11eb-9310-0b50250caeab.png" width="85%"//>
</p>

The ASR is based on [Wav2vec 2.0](https://arxiv.org/pdf/2006.11477.pdf) and is used to transcribe the speech from a source speaker. The TTS is based on [SV2TTS](https://arxiv.org/pdf/1806.04558.pdf) and is used to generate the output speech from a target speaker embedding.

For a more detailed explanation checkout [the paper of our project](https://github.com/fmiotello/fastVC/files/6711123/paper.pdf). A demo page is available [here](https://fmiotello.github.io/fastVC).

## Installation & usage

The software was implemented using `python 3.9.4`

1. Clone the repository (`git clone https://github.com/fmiotello/fastVC.git`) and enter the directory (`cd fastVC`)
2. (*optional*) Create virtual env and activate it: `python -m venv env` and `source env/bin/activate` (if using macOS/Linux) or `.\env\Scripts\activate` (if using Windows)
3. Upgrade pip: `python -m pip install --upgrade pip`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Download the [pretrained models](https://github.com/blue-fish/Real-Time-Voice-Cloning/releases/download/v1.0/pretrained.zip) and put them in the correct directories:
  ````
  ./src/encoder/saved_models/pretrained.pt
  ./src/synthesizer/saved_models/pretrained/pretrained.pt
  ./src/vocoder/saved_models/pretrained/pretrained.pt
  ````
6. Run the main script: `python src/main.py` (use `--help` for displaying available options). The output audio will be `./src/audio/audio_out.wav`.

More instructions can be found [here](https://github.com/fmiotello/fastVC/tree/main/src).

## Notes

This application was developed as a project at [Politecnico di Milano](https://www.polimi.it/en/) (MSc in Music and Acoustic Engineering).

*[Luigi Attorresi](https://github.com/LuigiAttorresi)*<br>
*[Federico Miotello](https://github.com/fmiotello)*<br>
*[Eugenio Poliuti](https://github.com/Poliuti)*<br>
