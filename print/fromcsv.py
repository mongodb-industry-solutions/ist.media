#
# Download .png files which are referenced in a csv file
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz


import pandas as pd
import requests
import os, re

file_path = 'graphics.csv'
data = pd.read_csv(file_path, header=None)

if data.shape[1] < 3:
    raise ValueError("CSV must have at least 3 columns")

output_dir = 'downloaded_pngs'
os.makedirs(output_dir, exist_ok=True)

# download PNGs from column 2, use column 3 for filenames
for index, row in data.iterrows():
    url = row[1]
    filename = row[2]
    # cut filenames
    filename = filename[:75]
    # replace all whitespace with underscores, then keep only allowed characters
    filename = re.sub(r'\s+', '_', filename)
    filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
    if not filename.lower().endswith('.png'):
        filename += '.png'
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(os.path.join(output_dir, filename), 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print("All downloads completed.")
