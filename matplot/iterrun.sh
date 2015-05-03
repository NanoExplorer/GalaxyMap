#!/bin/sh
FILES=./settings/*.json
for f in $FILES
do
    echo "Processing $f"
    #python galaxy.py divide $f
    python galaxy.py correlation $f
done

OUTPUTS=./output/*.json
for f in $OUTPUTS
do
    echo "Processing $f"
    python galaxy.py stats $f
done
