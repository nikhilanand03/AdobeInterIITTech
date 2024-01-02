# DEPENDENCIES
# !git clone https://github.com/PKU-YuanGroup/LanguageBind
# !cd LanguageBind
# !pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# #!pip install -r requirements.txt
# !pip install einops
# !pip install peft
# !pip install transformers==4.31.0
# !pip install decord
# !pip install pytorchvideo
# !pip install urlextract

from urlextract import URLExtract
# from google.colab import output
import urllib.request
import moviepy.editor
from moviepy.editor import VideoFileClip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

pretrained_ckpt = 'LanguageBind/LanguageBind_Video_FT'  # also 'LanguageBind/LanguageBind_Video'
model_v = LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
tokenizer_v = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
video_process = LanguageBindVideoProcessor(model_v.config, tokenizer_v)

pretrained_ckpt = 'LanguageBind/LanguageBind_Audio'  # also 'LanguageBind/LanguageBind_Audio'
model_a = LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
tokenizer_a = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
audio_process = LanguageBindAudioProcessor(model_a.config, tokenizer_a)

pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
model_i = LanguageBindImage.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
tokenizer_i = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
image_process = LanguageBindImageProcessor(model_i.config, tokenizer_i)

def encode_video(url):
  if('/vid/' not in url):
    print("not a video")
    return
  urllib.request.urlretrieve(url,"vid.mp4")
  print("Got video")
  data = video_process(["/content/vid.mp4"], ["text"], return_tensors='pt')
  print("Processed video")
  #data = {key: val.to(device) for key, val in data.items()}  # Move data to GPU
  model_v.eval()

  with torch.no_grad():
      out = model_v(**data)
      return out.image_embeds  # Return image embeddings
  print("Got embeddings")

def encode_image(url):
  if('/vid/' in url):
    print("not an image")
    return
  urllib.request.urlretrieve(url,"img.jpg")
  print("Got image")
  data = image_process(["/content/img.jpg"], ["text"], return_tensors='pt')
  print("Processed image")
  #data = {key: val.to(device) for key, val in data.items()}  # Move data to GPU
  model_i.eval()

  with torch.no_grad():
      out = model_i(**data)
      return out.image_embeds  # Return image embeddings
  print("Got embeddings")

def encode_audio(url):
  if('/vid/' not in url):
    print("not a video")
    return
  urllib.request.urlretrieve(url,"vid.mp4")
  print("Got video")

  video = moviepy.editor.VideoFileClip("vid.mp4")
  audio = video.audio
  audio.write_audiofile("audio.wav")
  print("Got audio")

  data = audio_process(["/content/audio.wav"], ["text"], return_tensors='pt')
  print("Processed audio")
  #data = {key: val.to(device) for key, val in data.items()}  # Move data to GPU
  model_a.eval()

  with torch.no_grad():
      out = model_a(**data)
      return out.image_embeds  # Return image embeddings
  print("Got embeddings")