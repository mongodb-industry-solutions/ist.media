import os
from pydub import AudioSegment
import openai

client = openai.OpenAI()

input_file = "tagesschau.mp3"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

audio = AudioSegment.from_mp3(input_file)
chunk_duration = 10 * 1000

for start_ms in range(0, len(audio), chunk_duration):
    chunk = audio[start_ms:start_ms + chunk_duration]
    offset_sec = start_ms // 1000
    chunk_filename = os.path.join(output_dir, f"temp_{offset_sec:04d}.mp3")
    chunk.export(chunk_filename, format="mp3")

    try:
        with open(chunk_filename, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="de"
            )
            text = transcript.text.strip()
    except Exception as e:
        text = f"[Fehler: {e}]"

    output_txt = os.path.join(output_dir, f"frame_{offset_sec:04d}.txt")
    with open(output_txt, "w") as out_f:
        out_f.write(text)

    os.remove(chunk_filename)
    print(f"Transkribiert: Sekunde {offset_sec:04d}")

print("Fertig.")
