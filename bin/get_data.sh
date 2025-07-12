if [ ! -d "data/external" ]; then
    mkdir -p data/external
fi

## THINGS ##
# spose
if [ ! -f "data/external/spose_embedding_66d_sorted.txt" ]; then
    wget --content-disposition -P data/external/ https://osf.io/nfbp3/download
fi

# spose labels
if [ ! -f "data/external/unique_id.txt" ]; then
    wget --content-disposition -P data/external/ https://osf.io/7zuah/download
fi

# description of dimensions
if [ ! -f "data/external/labels.txt" ]; then
    wget --content-disposition -P data/external/ https://osf.io/ydxus/download
fi

# things images
if [ ! -d "data/external/THINGS" ]; then
    wget -O data/external/THINGS.zip https://things-initiative.org/uploads/THINGS/images.zip
    unzip data/external/THINGS.zip -d data/external/THINGS
    rm data/external/THINGS.zip
fi

## COCO ##

# coco images
if [ ! -d "data/external/coco" ]; then
    wget -O data/external/coco.zip http://images.cocodataset.org/zips/train2017.zip
    unzip data/external/coco.zip -d data/external/coco
    rm data/external/coco.zip
fi