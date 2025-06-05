from moviepy import VideoFileClip
import os

# Path to the input video
video_path = 'tagesschau.mp4'
# Directory to save frames
output_dir = 'frames'
# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the video
clip = VideoFileClip(video_path)

# Get video duration and fps
duration = clip.duration
fps = clip.fps

# Extract frames at 2-second intervals
for t in range(0, int(duration), 2):
    frame = clip.get_frame(t)
    output_path = os.path.join(output_dir, f'frame_{t:04d}.jpg')
    clip.save_frame(output_path, t=t)

# Close the clip
clip.close()
