from moviepy import VideoFileClip

video = VideoFileClip("tagesschau.mp4")
video.audio.write_audiofile("tagesschau.mp3")
video.close()
