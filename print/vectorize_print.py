#
# Create vector embeddings for scanned media
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz
#
# Provided file format:
# - one directory per issue of the day (e.g., 19690704/ for July 4th 1969)
# - one file per page scan (e.g., FTDA-1969-0704-0021.JPG for page 21)

import os
import json
from PIL import Image
import voyageai

client = voyageai.Client()

media_folder = "./PrintMedia/"
embeddings_folder = "./PrintMediaEmbeddings/"
MAX_PIXELS = 16000000  # Voyage multimodal-3 limit

# Create embeddings folder if it doesn't exist
os.makedirs(embeddings_folder, exist_ok=True)

for date_dir in os.listdir(media_folder):
    date_path = os.path.join(media_folder, date_dir)
    if os.path.isdir(date_path):
        # Parse date from directory name (e.g., 19690725 -> 1969-07-25)
        if len(date_dir) == 8 and date_dir.isdigit():
            year = date_dir[:4]
            month = date_dir[4:6]
            day = date_dir[6:8]
            issue_date = f"{year}-{month}-{day}"

            # Create subfolder in embeddings
            date_embeddings_path = os.path.join(embeddings_folder, date_dir)
            os.makedirs(date_embeddings_path, exist_ok=True)

            for filename in sorted(os.listdir(date_path)):  # Sort for consistent page order
                if filename.endswith('.JPG'):
                    filepath = os.path.join(date_path, filename)

                    try:
                        # Parse page number from filename (e.g., FTDA-1969-0725-0037.JPG -> 37)
                        base_name = os.path.splitext(filename)[0]
                        page_parts = base_name.split('-')
                        page_str = page_parts[-1] if len(page_parts) > 1 and page_parts[-1].isdigit() else None
                        if page_str:
                            page = int(page_str)
                        else:
                            print(f"Could not parse page from {filename}, skipping.")
                            continue

                        # Open image as PIL object
                        img = Image.open(filepath)
                        original_size = img.size  # (width, height)
                        original_pixels = original_size[0] * original_size[1]

                        # Downscale if exceeding pixel limit (preserve aspect ratio)
                        processed_img = img
                        processed_size = original_size
                        scale_info = None
                        if original_pixels > MAX_PIXELS:
                            width, height = original_size
                            scale_factor = (MAX_PIXELS / original_pixels) ** 0.5
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            # Fine-tune to exact limit if needed (adjust larger dimension minimally)
                            while new_width * new_height > MAX_PIXELS:
                                if width > height:
                                    new_width -= 1
                                else:
                                    new_height -= 1
                            processed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            processed_size = (new_width, new_height)
                            processed_pixels = new_width * new_height
                            scale_info = {
                                "original_pixels": original_pixels,
                                "scale_factor": float(scale_factor),
                                "processed_pixels": processed_pixels
                            }
                            print(
                                f"Resized {filename}: "
                                f"{original_size} = {original_pixels} px -> "
                                f"{processed_size} = {processed_pixels} px"
                            )

                        # Image-only embedding: inputs = [[processed_img]]
                        inputs = [[processed_img]]
                        result = client.multimodal_embed(inputs, model="voyage-multimodal-3")
                        embedding = result.embeddings[0]

                        # Prepare data with full relative path
                        relative_path = f"PrintMedia/{date_dir}/{filename}"
                        page_data = {
                            "issue_date": issue_date,
                            "page": page,
                            "image_filename": relative_path,
                            "original_size": original_size,  # (width, height)
                            "processed_size": processed_size,  # For embedding (after resize if needed)
                            "scale_info": scale_info,  # Optional: Details if resized
                            "embedding": embedding
                        }

                        # Output JSON file
                        output_filename = f"embedding_{base_name}.json"
                        output_file = os.path.join(date_embeddings_path, output_filename)

                        with open(output_file, "w") as f:
                            json.dump(page_data, f)

                        print(f"Vectorized: {filename} -> {output_filename}")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

print("Done.")
