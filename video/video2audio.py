import sys, os
from moviepy import VideoFileClip

def main():
    if len(sys.argv) < 2:
        print("Usage: python video2audio.py <video.mp4>")
        sys.exit(1)

    mp4_file = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(mp4_file):
        print(f"File '{mp4_file}' not found.")
        sys.exit(1)

    print(f"Converting '{mp4_file}' to MP3...")

    # Load the video
    video = VideoFileClip(mp4_file)

    try:
        # Define target MP3 filename
        mp3_file = os.path.splitext(mp4_file)[0] + ".mp3"

        # Extract and save audio
        video.audio.write_audiofile(mp3_file)

        print(f"Saved MP3 as '{mp3_file}'")
    finally:
        # Clean up resources
        if video.audio is not None:
            video.audio.close()
        video.close()

if __name__ == "__main__":
    main()
