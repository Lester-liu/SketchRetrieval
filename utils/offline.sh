#!/usr/bin/env bash

Data="$1"

bash renderer.sh ../../../data/$Data/models_ply/ ../../../data/$Data/views/ 6
bash contour_extractor.sh ../../../data/$Data/views/ ../../../data/$Data/contours/
bash view_selector.sh ../../../data/$Data/views/ view.txt

