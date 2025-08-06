#
# Extract frames from mp4 video
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from moviepy import VideoFileClip
import os

video_path = 'video.mp4' # replace with your actual filename
output_dir = 'video.dir' # same here
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

clip = VideoFileClip(video_path)
duration = clip.duration
fps = clip.fps

for t in range(0, int(duration), 2):
    frame = clip.get_frame(t)
    output_path = os.path.join(output_dir, f'frame_{t:04d}.jpg')
    clip.save_frame(output_path, t=t)

clip.close()
