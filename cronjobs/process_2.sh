#!/bin/sh

. /usr/local/share/rt/bin/activate
cd /usr/local/share/ist.media/scripts
python create_images.py
cd /var/tmp/images.ist.media
for i in *.png; do
  magick "$i" "${i%.png}.webp"
done
cp *.webp /usr/local/share/content/images
rm *.webp *.png
cd /usr/local/share/ist.media/scripts
python process_incoming.py
python vectorize.py
python add_sentiment.py
python add_sections.py
