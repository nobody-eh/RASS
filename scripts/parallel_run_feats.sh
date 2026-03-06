#!/bin/bash

find $1 -mindepth 1 -maxdepth 1 -type d | parallel -I% --max-args 1 --jobs 6 --linebuffer bash scripts/feats.sh %