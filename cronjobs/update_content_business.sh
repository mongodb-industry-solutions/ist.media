#!/bin/sh

. /usr/local/share/rt/bin/activate
cd /usr/local/share/ist.media/scripts
python news_collect_2.py 10001147 # business
sh ../cronjobs/process_2.sh
