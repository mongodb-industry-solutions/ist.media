import os
import json
import numpy as np
import voyageai

client = voyageai.Client()

frame_folder = "./frames/"
query_text = "Bombe Köln Polizei"
threshold = 0.1

# JSON-Dateien laden
frames = []
for fname in sorted(os.listdir(frame_folder)):
    if fname.endswith(".json") and fname.startswith("frame_"):
        path = os.path.join(frame_folder, fname)
        with open(path, "r") as f:
            data = json.load(f)
            frames.append({
                "offset": data["offset"],
                "embedding": data["embedding"]
            })

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Query-Vektor erzeugen (Text-only)
query_embedding = np.array(client.multimodal_embed([[query_text]], model="voyage-multimodal-3").embeddings[0])

# Bester Frame (höchste Ähnlichkeit)
best_score = -1
best_offset = None
for frame in frames:
    score = cosine_similarity(query_embedding, frame["embedding"])
    if score > best_score:
        best_score = score
        best_offset = frame["offset"]

print(f"Bestes Ergebnis für '{query_text}': Offset {best_offset} mit Score {best_score:.4f}")


scores = []
sorted_frames = sorted(frames, key=lambda x: x["offset"])
for frame in sorted_frames:
    score = cosine_similarity(query_embedding, frame["embedding"])
    scores.append(score)

best_idx = np.argmax(scores)
scene_start = None
prev_score = scores[best_idx]

for i in range(best_idx, -1, -1):
    score = scores[i]
    if prev_score - score > 0.01:
        scene_start = sorted_frames[i]["offset"]
        break
    prev_score = score

if scene_start is None:
    scene_start = sorted_frames[0]["offset"]


print(best_idx)

if scene_start is not None:
    print(f"Szenenanfang bei Offset {scene_start}s (Score ≥ {threshold})")
else:
    print(f"Keine Szene mit Score ≥ {threshold} gefunden.")
