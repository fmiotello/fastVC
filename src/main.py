### IMPORTS ###

print('Starting imports...')

import soundfile as sf
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset

print('...imports done')


### MODELS DOWNLOAD ###

print('Downloading models...')

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

print('...download done')


### MAIN ###

SAMPLE_RATE = 16000

input_audio, sr = librosa.load("audio/audio_in.wav", sr=SAMPLE_RATE)
input_values = tokenizer(np.asarray(input_audio), return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

print(transcription)

