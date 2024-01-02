# SETUP:
# BRANCH = 'r1.21.0'
# !python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
# Import NeMo and it's ASR, NLP and TTS collections

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tts
import IPython
import urllib.request
import moviepy.editor
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

def get_audio(url):
  if('/vid/' not in url):
    print("not a video")
    return
  urllib.request.urlretrieve(url,"vid.mp4")
  print("Got video")

  video = moviepy.editor.VideoFileClip("vid.mp4")
  audio = video.audio
  audio.write_audiofile("audio.wav")
  print("Got audio")

def extractSpeechText(url):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    get_audio(url)
    input_file = "audio.wav"
    output_file = "output.wav"
    audio = AudioSegment.from_wav(input_file)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(output_file, format="wav")
    print(f"Conversion complete. Output file: {output_file}")
    transcribed_text = asr_model.transcribe(["output.wav"])
    return transcribed_text