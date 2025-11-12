#!/bin/sh

. /usr/local/share/rt/bin/activate
cd /usr/local/share/ist.media/scripts

case "$1" in
    autos)
        FEED_ID="10000101"
        ;;
    business)
        FEED_ID="10001147"
        ;;
    earnings)
        FEED_ID="15839135"
        ;;
    europe)
        FEED_ID="19794221"
        ;;
    finance)
        FEED_ID="10000664"
        ;;
    intnews)
        FEED_ID="100727362"
        ;;
    tech)
        FEED_ID="19854910"
        ;;
    travel)
        FEED_ID="10000739"
        ;;
    usnews)
        FEED_ID="15837362"
        ;;
    *)
        echo "Error: Unknown section '$1'"
        exit 1
        ;;
esac

python news_collect_2.py "$FEED_ID"
sh ../cronjobs/process_2.sh
