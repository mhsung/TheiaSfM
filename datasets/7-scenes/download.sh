# Minhyuk Sung (mhsung@cs.stanford.edu)

# Download
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip

# Unzip
unzip '*.zip'
for dataset in *; do
	if [ -d "${dataset}" ]; then
		unzip ${dataset}/'*.zip' -d ${dataset}
		rm ${dataset}/*.zip
	fi
done

# Move zip files to downloads directory
if [ ! -d "downloads" ]; then
	mkdir downloads
fi
mv *.zip downloads
