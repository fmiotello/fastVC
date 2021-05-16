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

print('[...imports done]')


### MODELS DOWNLOAD ###

print('[Loading models...]')

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

encoder.load_model(Path("src/encoder/saved_models/pretrained.pt"))
synthesizer = Synthesizer(Path("src/synthesizer/saved_models/pretrained/pretrained.pt"))
vocoder.load_model(Path("src/vocoder/saved_models/pretrained/pretrained.pt"))

print('[...load done]')

### MAIN ###

SAMPLE_RATE = 16000

input_audio, sr = librosa.load("audio/audio_in.wav", sr=SAMPLE_RATE)
input_values = tokenizer(np.asarray(input_audio), return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

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
  print('[...synthesis done]')
  return generated_wav

SAMPLE_RATE = 16000
text = listToString(transcription)
print('Detected sentence: ' + text)
embedding = encoder.embed_utterance(encoder.preprocess_wav(input_audio, SAMPLE_RATE))

out_audio = synthesize(embedding, text)
sf.write('audio/audio_out.wav', out_audio, 16000)

print('[DONE]')
