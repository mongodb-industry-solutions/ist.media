#!/bin/sh

. /usr/local/share/rt/bin/activate
cd /usr/local/share/ist.media/scripts
python news_collect_2.py 100727362 # intl news
sh ../../process_2.sh
