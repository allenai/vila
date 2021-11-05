#!/bin/bash

dataset_name="$1"
base_save_path="../data"
mkdir -p $base_save_path

S3_BASE_LINK="https://ai2-s2-research.s3.us-west-2.amazonaws.com/s2-vlue"
GROTOAP2_S3_NAME="grotoap2.zip"
DOCBANK_S3_NAME="docbank.zip"
S2_VL_VER1_S3_NAME="s2-vl-ver1-public.zip"

download_complied_dataset () {
    target_path="$1"
    s3_name="$2"
    wget $S3_BASE_LINK/$s3_name -O $base_save_path/$s3_name
    unzip $base_save_path/$s3_name -d $base_save_path/$target_path  
    rm $base_save_path/$s3_name
}

case $dataset_name in

  grotoap2)
    download_complied_dataset "grotoap2" $GROTOAP2_S3_NAME
    ;;

  docbank)
    download_complied_dataset "docbank" $DOCBANK_S3_NAME
    ;;

  s2-vl)
    download_complied_dataset "s2-vl-ver1" $S2_VL_VER1_S3_NAME
    ;;

  all)
    download_complied_dataset "grotoap2" $GROTOAP2_S3_NAME
    download_complied_dataset "docbank" $DOCBANK_S3_NAME
    download_complied_dataset "s2-vl-ver1" $S2_VL_VER1_S3_NAME
    ;;

  *)
    echo -n "Unkown Dataset"
    exit
    ;;
esac