#
# Create multimodal vector files
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

import os
import json
from PIL import Image
import voyageai

client = voyageai.Client()
image_folder = "./video.dir/" # replace with your actual directory name

for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")) and filename.startswith("frame_"):
        filepath = os.path.join(image_folder, filename)
        base = os.path.splitext(filename)[0]
        offset_str = base.split("_")[1]
        offset = int(offset_str)

        text_offset = (offset // 20) * 20
        text_file = os.path.join(image_folder, f"frame_{text_offset:04d}.txt")

        if not os.path.isfile(text_file):
            print(f"No text for second {offset} → {text_file} is missing")
            continue

        try:
            with open(text_file, "r") as tf:
                transcript = tf.read().strip()

            img = Image.open(filepath)
            inputs = [[transcript, img]]
            result = client.multimodal_embed(inputs, model="voyage-multimodal-3")
            embedding = result.embeddings[0]

            frame_data = {
                "movie": "video", # replace with your actual movie/video name
                "offset": offset,
                "text_offset": text_offset,
                "embedding": embedding
            }

            output_file = os.path.join(image_folder, f"{base}.json")
            with open(output_file, "w") as f:
                json.dump(frame_data, f)

            print(f"Vectorized: {filename} with text {text_file}")

        except Exception as e:
            print(f"Error at {filename}: {e}")

print("All JSON files have been created.")
