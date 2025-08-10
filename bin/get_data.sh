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
    unzip data/external/THINGS.zip -d data/external/
    mv data/external/images data/external/THINGS
    rm data/external/THINGS.zip
fi

# things odd-one-out
if [ ! -d "data/external/THINGS_odd_one_out" ]; then
    wget -O data/external/THINGS_odd_one_out.zip https://osf.io/n9u4a/download
    unzip data/external/THINGS_odd_one_out.zip -d data/external/
    mv data/external/triplet_dataset data/external/THINGS_odd_one_out
    rm data/external/THINGS_odd_one_out.zip
fi

## STUFF ##

# stuff odd-one-out
if [ ! -d "data/external/STUFF_odd_one_out" ]; then
    mkdir -p data/external/STUFF_odd_one_out
    wget -O data/external/STUFF_odd_one_out/train_90.txt https://osf.io/m437z/download
    wget -O data/external/STUFF_odd_one_out/test_10.txt https://osf.io/35qsb/download

fi

# stuff images
if [ ! -d "data/external/STUFF" ]; then
    wget -O data/external/stuff.zip https://osf.io/b73xv/download
    unzip -P stuff4all data/external/stuff.zip -d data/external/
    mv data/external/STUFF_dataset_600_images data/external/STUFF
    rm data/external/stuff.zip
fi

## COCO ##

# coco images
if [ ! -d "data/external/coco/train2017" ]; then
    wget -O data/external/coco_train.zip http://images.cocodataset.org/zips/train2017.zip
    unzip data/external/coco_train.zip -d data/external/
    mv data/external/train2017 data/external/coco/
    rm data/external/coco_train.zip
fi

if [ ! -d "data/external/coco/val2017" ]; then
    wget -O data/external/coco_val.zip http://images.cocodataset.org/zips/val2017.zip
    unzip data/external/coco_val.zip -d data/external/
    mv data/external/val2017 data/external/coco/
    rm data/external/coco_val.zip
fi

if [ ! -d "data/external/coco/test2017" ]; then
    wget -O data/external/coco_test.zip http://images.cocodataset.org/zips/test2017.zip
    unzip data/external/coco_test.zip -d data/external/
    mv data/external/test2017 data/external/coco/
    rm data/external/coco_test.zip
fi