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
image_folder = "./Charts/" # replace with your actual directory name

for i, filename in enumerate(sorted(os.listdir(image_folder)), 1):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filepath = os.path.join(image_folder, filename)
        description = os.path.splitext(filename)[0].replace('_', ' ')
        json_file = os.path.join(image_folder, f"v_{i}.json")

        try:
            img = Image.open(filepath)
            inputs = [[description, img]]
            result = client.multimodal_embed(inputs, model="voyage-multimodal-3")
            embedding = result.embeddings[0]
            json_data = {
                "description": description,
                "embedding": embedding
            }
            with open(json_file, "w") as f:
                json.dump(json_data, f)

            print(f"Vectorized: {filename}")

        except Exception as e:
            print(f"Error at {filename}: {e}")

print("All JSON files have been created.")
