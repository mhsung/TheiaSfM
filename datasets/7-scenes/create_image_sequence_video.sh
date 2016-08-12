# Minhyuk Sung (mhsung@cs.stanford.edu)

if [ "$#" -lt 1 ]; then
	echo "Required arguments: "
	echo " - Data path"
	exit
fi

data_dir=$1
data_name=$(basename "${data_dir}")
echo $1
echo ${data_name}

if [ ! -d ${data_dir} ]; then
	echo "Data path does not exist: '${data_dir}'"
	exit
fi

if [ ! -d ${data_dir}/../videos ]; then
	mkdir ${data_dir}/../videos
fi

ffmpeg -i ${data_dir}/images/frame-%06d.png\
	${data_dir}/../videos/${data_name}.mp4

