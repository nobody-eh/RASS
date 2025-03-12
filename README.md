# VolETA-v4
A near-real time video food volume estimation


find /media/amughrabi/mypassport/data/MetaFood3D_new_RGBD_videos/RGBD_videos -type d -name depths -exec dirname {} \; | parallel -I? --max-args 1 --jobs 12 --linebuffer bash scripts/compute_depths.sh
