if [ ! -d "data/external" ]; then
    mkdir -p data/external
fi

if [ ! -f "data/external/spose_embedding_66d_sorted.txt" ]; then
    wget --content-disposition -P data/external/ https://osf.io/nfbp3/download
fi

if [ ! -f "data/external/unique_id.txt" ]; then
    wget --content-disposition -P data/external/ https://osf.io/7zuah/download
fi

if [ ! -d "data/external/THINGS" ]; then
    wget -O data/external/THINGS.zip https://things-initiative.org/uploads/THINGS/images.zip
    unzip data/external/THINGS.zip -d data/external/THINGS
    rm data/external/THINGS.zip
fi