#!/bin/bash

# Some images contains high amount of features (i.e. rich scene background objects: bike, ball, textured, etc) to be
# paralleled. Since it is fully utilise the GPU (100%), we make it sequential. If you see your data features can be
# paralleled, increase --jobs parameter with the suitable number based on your GPU.
#find $1 -mindepth 1 -maxdepth 1 -type d | parallel -I% --max-args 1 --jobs 1 --linebuffer bash scripts/run_ingp.sh %/t

dirname="$1"
cont=$(cat "$dirname")
echo "Contingut del fitxer:"
fitxers="directori_clustered.txt"
> "$fitxers"
#echo "$cont"
for dish in $cont; do
    if [ "$dish" = "dish_id" ]; then
        continue
    fi
    fitxer=$(find n5k360p/ -type d -name  "$dish")

    echo "$fitxer" >> "$fitxers"
    #echo "Processing $dish/test"
    #echo "$fitxer"
done

cat "$fitxers" | parallel --jobs 1 --linebuffer bash scripts/run_ingp.sh {}/images

#rm -f "$fitxers"
