# Minhyuk Sung (mhsung@cs.stanford.edu)

find . -name "seq-*" -exec echo {} \;
find . -name "seq-*" -exec ./organize_data.sh {} \;
find . -name "seq-*" -exec ./convert_pose_files.py {} \;
find . -name "seq-*" -exec ./create_image_sequence_video.sh {} \;
