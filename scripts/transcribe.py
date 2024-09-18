from moviepy.editor import VideoFileClip
from PIL import Image
import os, whisper


def extract_audio(video_file, output_audio_file):
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(output_audio_file)


def transcribe_audio(audio_file):
    model = whisper.load_model("small") # 'base', 'small', 'medium', 'large'
    result = model.transcribe(audio_file)
    return result['text']


def transcribe_video(mp4_file):
    audio_file = "/Users/bjjl/audio.wav"
    extract_audio(mp4_file, audio_file)
    transcript = transcribe_audio(audio_file)
    return transcript

transcript = transcribe_video("/Users/bjjl/video.mp4")
print(transcript)

#########
#########



def extract_frames(video_file, output_folder, interval=10):
    clip = VideoFileClip(video_file)
    duration = clip.duration 
    times = range(0, int(duration), interval) 
    
    for t in times:
        frame = clip.get_frame(t)
        output_path = f"{output_folder}/frame_{t}.jpg"
        frame_image = Image.fromarray(frame)
        frame_image.save(output_path)

video_file = "/Users/bjjl/video.mov"
output_folder = "/Users/bjjl/frames" 

extract_frames(video_file, output_folder, interval=30)
