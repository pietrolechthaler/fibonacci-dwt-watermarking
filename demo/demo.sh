#!/bin/bash

##### Demo code for for watermark-challenge code

echo "Starting demo..."
cd src

echo "----------------------------"
echo "EMBEDDING AND DECODING PHASE"
python3 launcher.py ../demo/demo_images

echo "----------------------------"
echo "ATTACKING PHASE"
python3 attacks.py ../demo/demo_images/0000.bmp ../watermarked_images/polymer_0000.bmp polymer

echo "Demo finished, check the results of your attack inside src/results/polymer"