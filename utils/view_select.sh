#!/usr/bin/env bash

#!/usr/bin/env bash

# One folders with PNG format files and one folder to the contour

Files="$1"

for Folder in $Files*/
do
    ./sample/Debug/./sample $Folder $Folder/view.txt
done