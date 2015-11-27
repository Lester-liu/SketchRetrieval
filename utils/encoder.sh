#!/usr/bin/env bash

#Ex: bash encoder.sh ../../../data/Sketch/SHREC13_SBR_Model_Index.txt ../../../data/TinySketch/views/ view.txt ../../../data/TinySketch/contours/ ../../../data/TinySketch/result.dict ../../../data/TinySketch/encodes/ 36 784

Indexer="$1" # file containing all models' name
View_Path="$2" # folder to views (containing the selector result file)
View_File="$3" # selector result name
Contour_Path="$4" # path to contour folder
Dict="$5"; # dictionary file
Encode_Folder="$6" # path to output folder
Cases="$7" # case number (select some images)
Size="$8" # number of local features per image

while read -r Line
do
    Model=${Line%%[[:space:]]} # trailing end of line
    #echo "./KMean/Release/k_mean -4 -f $View_Path$Model/$View_File -r $Contour_Path$Model/ -d $Dict -s 32 -o $Encode_Folder$Model.trans -c $Cases -a $Size" > /home/lyx/test
    ./KMean/Release/k_mean -4 -f $View_Path$Model/$View_File -r $Contour_Path$Model/ -d $Dict -s 32 -o $Encode_Folder$Model.trans -c $Cases -a $Size
done < "$Indexer"
