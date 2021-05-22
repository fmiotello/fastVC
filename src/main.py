### IMPORTS ###

print('[Starting imports...]')

import soundfile as sf
import torch
import librosa
import numpy as np
import sys
import os
from os.path import exists, join, basename, splitext
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import argparse
from utils.argutils import print_args

print('[...imports done]')


### MODELS DOWNLOAD ###

print('[Loading models...]')

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

encoder.load_model(Path("src/encoder/saved_models/pretrained.pt"))
synthesizer = Synthesizer(Path("src/synthesizer/saved_models/pretrained/pretrained.pt"))
vocoder.load_model(Path("src/vocoder/saved_models/pretrained/pretrained.pt"))

print('[...load done]')

### FUNCTIONS and GLOBAL VARIABLES ###

SAMPLE_RATE = 16000

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

def synthesize(embed, text):
  print('[Synthesizing new audio...]')
  #with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
  print('\n[...synthesis done]')
  return generated_wav

### MAIN ###

if __name__ == '__main__':

    # args parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=None, help=\
        "Source speaker for voice conversion.")
    parser.add_argument("--target", type=Path, default=None, help=\
        "Target speaker for voice conversion.")
    parser.add_argument("--string", type=str, default=None, help=\
        "Sentence used by the synthesized voice.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")

    args = parser.parse_args()
    print_args(args, parser)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)

    if args.source is not None:
        source_audio, _ = librosa.load(args.source, sr=SAMPLE_RATE)
    else:
        source_audio, _ = librosa.load("audio/source.wav", sr=SAMPLE_RATE)

    if args.target is not None:
        target_audio, _ = librosa.load(args.target, sr=SAMPLE_RATE)
    else:
        target_audio, _ = librosa.load("audio/target.wav", sr=SAMPLE_RATE)

    if args.string is not None:
        if args.source is not None:
            raise Exception("[ERROR] Can't specify both source and string args.")
        text = args.string
    else:
        input_values = tokenizer(np.asarray(source_audio), return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)
        text = listToString(transcription)

    print('Detected sentence: ' + text)

    embedding = encoder.embed_utterance(encoder.preprocess_wav(target_audio, SAMPLE_RATE))

    out_audio = synthesize(embedding, text)
    sf.write('audio/audio_out.wav', out_audio, 16000)
