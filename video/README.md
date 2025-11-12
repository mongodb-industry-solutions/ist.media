# Helper Scripts for Video Search 

This directory contains the scripts that you can use to
pre-process all videos that shall participate in the demo.

I recommend to use media that is **not longer than 10 minutes**
in length, and ideally you want to **mix multiple languages**, as
this shows the power of language-independent, semantic search.

Videos shall be in **mp4 format** to be able to immediately use the
scripts below.

## video2audio.py

As the goal is to create multimodal embeddings for both voice track
and movie picture frames, this first step in the pre-processing is to
generate a pure voice document (i.e. mp3 file) that contains the **audio
channel of the video**.

## audio2text.py

From that mp3 file, we now generate **text fragments that can be
vectorized** each, to participate in the video search later.

In order to come up
with a good compromise between **meaningful embeddings** on the
one side, 
and **fine-granular
capability** to find the right time offset in the video on the other side,
I have decided to cut
the voice track into **20-second chunks**. This allows each chunk to contain
one or even a few complete sentences. You might want to experiment with
different chunk lengths, depending on the type of videos used. 

These chunks are then fed into a service that is turning the mp3 into
actual (UTF-8) text. I use whisper-1 from OpenAI for that, but there's plenty 
of other options how to extract text from audio. Choose what makes most
sense for your project or experiments. 

## extract_frames.py

When it comes to the pre-processing of the **picture channel**, the video
needs to be cut into picture frames that can be vectorized each. Also here, in order to
have **enough granularity** for finding a scene's start, the script needs to use
a pretty short time interval
between frames. By default it creates a frame  **every 2 seconds in the video**. 

Using more frames, e.g. every second, or even 500 ms, will blow up the search index
and not add much more precision. 
Using less frames will make it hard to successfully seek to
the beginning of a scene (it will be a bit too early, or a bit too late instead, 
for most of the searches).

## vectorize_multimodal.py

This final script is doing the actual vectorization, leveraging one of the most
advanced Voyage AI models, ``voyage-multimodal-3``. It is iterating over all files
that have been created, building tuples for joint vectorization. With the default
time intervalls (20 second voice track segments, picture every 2 seconds) this means
to that there are 10 vectors created per voice track segment, each time using one of 
the pictures within the 20 second boundary.

The resulting vectors, together with the meta data (which in essence are the
video name, and the voice / picture time offsets) can then be stored in
MongoDB Atlas, where the vectors are indexed, to become searchable with
Atlas Vector Search.


