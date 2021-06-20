The implementation of this tool is based on [this project](https://github.com/CorentinJ/Real-Time-Voice-Cloning), that implements the three-stage deep learning TTS framework proposed in [SV2TTS](https://arxiv.org/pdf/1806.04558.pdf). We extended it adding Facebook AI [Wav2vec 2.0](https://arxiv.org/pdf/2006.11477.pdf) ASR model.

For the computation of relevant metrics we used some classic NLP libraries: [NLTK](https://www.nltk.org), [JiWER](https://github.com/jitsi/jiwer), [speechmetrics](https://github.com/aliutkus/speechmetrics) and [asrtoolkit](https://github.com/finos/greenkey-asrtoolkit)

## Source code organization
`main.py` is the main script. This can be executed as a command line tool using `python main.py` (or `src/main.py` from outside this directory) with some options:
- `--source`: specify source speaker path (otherwise will use the default source speaker located in `./audio`
- `--target`: specify target speaker path (otherwise will use the default target speaker located in `./audio`
- `--string`: choose the string to give as input to the tts 
- `--seed`: random number seed to make the output deterministic
- `--metrics`: print relevant metrics
- `--enhance`: trim output audio silences
- `--help`: help page

`./audio` contains the default source and target speaker files and the output speech `audio_out.wav`.

`./encoder` contains the speaker encoder which creates a numerical representation of a voice from a few seconds of audio.

`./synthesizer` contains a modified text-to-speech synthesizer that generates an audio spectrogram in the target voice.

`./vocoder` contains the vocoder that transforms the spectrograms into waveform audio.

`./utils` contains some general utilities.
