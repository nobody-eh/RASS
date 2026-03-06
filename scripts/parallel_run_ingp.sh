#!/bin/bash


#find "$1" -mindepth 1 -maxdepth 1 -type d | parallel -I% --max-args 1 --jobs 1 --linebuffer  bash scripts/run_ingp.sh %/test 

#cat "$1" | parallel --jobs 1 --linebuffer bash scripts/run_ingp.sh {}/images

#find n5k360p/ -type d -name "$1" | parallel -I% --max-args 1 --jobs 1 --linebuffer  bash scripts/run_ingp.sh %/test 
#find n5k360p/ -type d -name dish_1565119464


dirname="$1"
cont=$(cat "$dirname")
echo "Contingut del fitxer:"
fitxers="directori_clustered.txt"
> "$fitxers"
#echo "$cont"  
for dish in $cont; do
    # Saltar la primera línia si és 'dish_id'
    if [ "$dish" = "dish_id" ]; then
        continue
    fi
    fitxer=$(find n5k360p/ -type d -name  "$dish")
    
    echo "$fitxer" >> "$fitxers"
    #echo "Processing $dish/test"
    echo "$fitxer"
done

cat "$fitxers" | parallel --jobs 1 --linebuffer bash scripts/run_ingp.sh {}/images

