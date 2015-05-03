#!/bin/sh
OUTPUTS=./output/*.json
for f in $OUTPUTS
do
    echo "Processing $f"
    python galaxy.py stats $f
done
