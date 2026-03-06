#!/bin/bash



# find $1 -mindepth 1 -maxdepth 1 -type d | parallel -I% --max-args 1 --jobs 8 --linebuffer  mkdir -p "%"/sparse/txt 


# find $1 -mindepth 1 -maxdepth 1 -type d | parallel -I% --max-args 1 --jobs 8 --linebuffer colmap model_converter --input_path "%"/sparse/0 --output_path "%"/sparse/txt --output_type TXT


find $1 -mindepth 1 -maxdepth 1 -type d | parallel -I% --max-args 1 --jobs 8 --linebuffer python3 src/colmap2nerf.py --colmap_db "%"/database.db --colmap_camera_model SIMPLE_RADIAL --images "%" --text "%"/sparse/txt --out "%"
