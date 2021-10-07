#!/bin/bash

dataset_name="$1"

GROTOAP2_BASELINK="https://ai2-s2-research.s3.us-west-2.amazonaws.com/scienceparseplus/datasets/grotoap2/grouped-v2"
DOCBANK_BASELINK="https://ai2-s2-research.s3.us-west-2.amazonaws.com/scienceparseplus/datasets/docbank/grouped-v6"

download_complied_dataset () {
    target_path="$1"
    base_link="$2"
    mkdir -p $target_path
    for file in "dev-token.json" "train-token.json" "test-token.json" "train-test-split.json" "labels.json"
    do
        wget $base_link/$file -O $target_path/$file
    done
}

case $dataset_name in

  grotoap2)
    download_complied_dataset "../data/grotoap2" $GROTOAP2_BASELINK
    ;;

  docbank)
    download_complied_dataset "../data/docbank" $DOCBANK_BASELINK
    ;;

  s2-vl)
    echo "WIP"
    ;;

  all)
    download_complied_dataset "../data/grotoap2" $GROTOAP2_BASELINK
    download_complied_dataset "../data/docbank" $DOCBANK_BASELINK
    ;;

  *)
    echo -n "Unkown Dataset"
    exit
    ;;
esac