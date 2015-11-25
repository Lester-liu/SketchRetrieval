#!/usr/bin/env bash

#!/usr/bin/env bash

# One folders with PNG format files and the name of the output file

Files="$1"
Name="$2"
for Folder in $Files*/
do
    ./view_selection/Debug/view_selection $Folder $Folder/$Name
done
