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
if [ ! -f "data/external/THINGS_triplets.csv" ]; then
    wget -O data/external/THINGS_odd_one_out.zip https://osf.io/n9u4a/download
    unzip data/external/THINGS_odd_one_out.zip -d data/external/
    mv data/external/triplet_dataset data/external/THINGS_odd_one_out
    rm data/external/THINGS_odd_one_out.zip
    mv data/external/THINGS_odd_one_out/triplets_large_final_correctednc_correctedorder.csv data/external/THINGS_triplets.csv
    rm -r data/external/THINGS_odd_one_out/
    rm -r __MACOSX/
fi

# things category learning
if [ ! -f "data/external/category_learning.csv" ]; then
    wget -O data/external/category_learning.csv https://osf.io/rsd46/download
fi

# things reward learning
if [ ! -f "data/external/reward_learning.csv" ]; then
    wget -O data/external/reward_learning.csv https://osf.io/6exjm/download
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

## Levels ##
if [ ! -f "data/external/levels.pkl" ]; then
    wget -O data/external/levels.pkl https://gin.g-node.org/fborn/Dataset_Levels/raw/master/processed_data/pruned_dataset.pkl
fi
## ImageNet ##
if [ ! -d "data/external/imagenet" ]; then
    wget -O data/external/ILSVRC2012_img_train.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
    wget -O data/external/ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
    cd data/external
    wget -qO- https://raw.githubusercontent.com/pytorch/examples/main/imagenet/extract_ILSVRC.sh | bash
fi