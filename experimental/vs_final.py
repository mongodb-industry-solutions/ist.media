import os
import json
from PIL import Image
import voyageai

client = voyageai.Client()
image_folder = "./wwdc2025.dir/"

for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")) and filename.startswith("frame_"):
        filepath = os.path.join(image_folder, filename)
        base = os.path.splitext(filename)[0]
        offset_str = base.split("_")[1]
        offset = int(offset_str)

        text_offset = (offset // 20) * 20
        text_file = os.path.join(image_folder, f"frame_{text_offset:04d}.txt")

        if not os.path.isfile(text_file):
            print(f"Kein Text gefunden für Sekunde {offset} → {text_file} fehlt")
            continue

        try:
            with open(text_file, "r") as tf:
                transcript = tf.read().strip()

            img = Image.open(filepath)
            inputs = [[transcript, img]]
            result = client.multimodal_embed(inputs, model="voyage-multimodal-3")
            embedding = result.embeddings[0]

            frame_data = {
                "movie": "wwdc2025",
                "offset": offset,
                "text_offset": text_offset,
                "embedding": embedding
            }

            output_file = os.path.join(image_folder, f"{base}.json")
            with open(output_file, "w") as f:
                json.dump(frame_data, f)

            print(f"Vektorisiert: {filename} mit Text {text_file}")

        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

print("Alle JSON-Dateien wurden erstellt.")
