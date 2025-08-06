#
# Create mp3 voice track from mp4 video
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

import sys, os
from moviepy import VideoFileClip

def main():
    if len(sys.argv) < 2:
        print("Missing input file")
        sys.exit(1)

    mp4_file = sys.argv[1]
    if not os.path.isfile(mp4_file):
        print(f"File '{mp4_file}' not found.")
        sys.exit(1)

    print(f"Converting '{mp4_file}' to mp3...")
    video = VideoFileClip(mp4_file)
    try:
        mp3_file = os.path.splitext(mp4_file)[0] + ".mp3"
        video.audio.write_audiofile(mp3_file)
        print(f"Saved mp3 as '{mp3_file}'")
    finally:
        if video.audio is not None:
            video.audio.close()
        video.close()

if __name__ == "__main__":
    main()
