# Minhyuk Sung (mhsung@cs.stanford.edu)

if [ "$#" -lt 1 ]; then
	echo "Required arguments: "
	echo " - Data path"
	exit
fi

data_dir=$1
if [ ! -d ${data_dir} ]; then
	echo "Data path does not exist: '${data_dir}'"
	exit
fi

echo ${data_dir}
rm ${data_dir}/Thumbs.db

mkdir ${data_dir}/images
mv ${data_dir}/*.color.png ${data_dir}/images
rename 's/\.color//g' ${data_dir}/images/*

mkdir ${data_dir}/depths
mv ${data_dir}/*.depth.png ${data_dir}/depths
rename 's/\.depth//' ${data_dir}/depths/*

mkdir ${data_dir}/poses
mv ${data_dir}/*.pose.txt ${data_dir}/poses
rename 's/\.pose//' ${data_dir}/poses/*

