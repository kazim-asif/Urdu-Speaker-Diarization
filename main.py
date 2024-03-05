
!pip install pydub
!pip install sentencepiece
!pip install replicate
!pip install googletrans
!pip install translate

!pip install yt_dlp

import yt_dlp

def download_video_and_get_title(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'video.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=True)  # Download the video and extract its info
        title = video_info.get('title', None)  # Get video title

    return title

download_video_and_get_title("https://www.youtube.com/watch?v=JR-J_4d2_0A")

!pip install moviepy

from moviepy.video.io.VideoFileClip import VideoFileClip

clip = VideoFileClip("video.mp4")
audio = clip.audio
audio.write_audiofile("audio.wav")
path = "audio.wav"

# upload audio file
from google.colab import files
uploaded = files.upload()
path = next(iter(uploaded))

num_speakers = 2

language = 'Urdu'

model_size = 'large'

model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

!pip install -q git+https://github.com/openai/whisper.git > /dev/null
!pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null

import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

if path[-3:] != 'wav':
  subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
  path = 'audio.wav'

model = whisper.load_model(model_size)

result = model.transcribe(path)
segments = result["segments"]

with contextlib.closing(wave.open(path,'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

audio = Audio()

def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)

    # Convert waveform to single channel
    waveform = waveform.mean(dim=0, keepdim=True)

    return embedding_model(waveform.unsqueeze(0))

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)


embeddings = np.nan_to_num(embeddings)

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
  return datetime.timedelta(seconds=round(secs))

f = open("transcript.txt", "w",encoding="UTF-8")

for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
  f.write(segment["text"][1:] + ' ')
f.close()

print(open('transcript.txt','r',encoding="UTF-8").read())

# from transformers import AutoProcessor, SeamlessM4TModel
# processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
# model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

# def translate(text_to_translate, src_lang, tgt_lang):
#   text_inputs = processor(text = text_to_translate, src_lang=src_lang, return_tensors="pt")
#   output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
#   translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
#   return translated_text

from translate import Translator

def translate(text, source_language, target_language):
    translator = Translator(to_lang=target_language, from_lang=source_language)
    translation = translator.translate(text)
    return translation

# Create dialogues directly from the segments
# dialogues = [f"{segment['speaker']}: {segment['text']}" for segment in segments]

# create english translated dialouges directly
dialogues = [f"{segment['speaker']}: {translate(segment['text'], 'ur', 'en')}" for segment in segments]

pip install replicate

from getpass import getpass
import os
import replicate

# Set the Replicate API token using environment variable
os.environ["REPLICATE_API_TOKEN"] = "Api token here"

# Function to generate prompt for the entire set of dialogues
def process_diarization(dialogues, pre_prompt):
    # Use a secure method to handle API tokens (e.g., environment variables)
    api_token = os.environ.get("REPLICATE_API_TOKEN")

    if not api_token:
        api_token = getpass("Enter your Replicate API token: ")

    # Combine all dialogues into a single text
    all_dialogues = "\n".join(dialogues)

    # Construct the prompt
    prompt = f"{pre_prompt}\n{all_dialogues}\n Assistant: "

    # Make sure the model and prompt are appropriate for Laama2 summarization
    model_id = "laama2 key here from replicate"

    # Run the model
    iterator_output = replicate.run(model_id, input={"prompt": prompt})

    combined_text = ""
    for text in iterator_output:
        combined_text += text  # combines the output

    return combined_text.strip()  # Strip to remove trailing spaces

pre_prompt = 'Below is the conversation between speakers. Please summarize the whole conversation:'

# Process and summarize the entire set of dialogues
summary = process_diarization(dialogues, pre_prompt)

# Print or use the summary as needed
print(summary)

translatedSummary = translate(summary, 'en', 'ur')
print(translatedSummary)
