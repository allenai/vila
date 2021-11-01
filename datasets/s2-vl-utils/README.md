# Recreating the S2-VL Dataset 

- [Recreating the S2-VL Dataset](#recreating-the-s2-vl-dataset)
  - [STEP0: Download the papers](#step0-download-the-papers)
  - [STEP1: Parse token data using CERMINE](#step1-parse-token-data-using-cermine)
  - [STEP2: Run visual layout detectors for getting the text block and line blocks](#step2-run-visual-layout-detectors-for-getting-the-text-block-and-line-blocks)
  - [STEP3: Assemble the annotations and export the dataset](#step3-assemble-the-annotations-and-export-the-dataset)

## STEP0: Download the papers 

TBD

## STEP1: Parse token data using CERMINE 

1. Download JAVA and CERMINE following instructions in [this repo](https://github.com/CeON/CERMINE#using-cermine) (PS: The easiest approach would be just downloading CERMINE v1.13 from [JFrog](http://maven.ceon.pl/artifactory/webapp/#/artifacts/browse/simple/General/kdd-releases/pl/edu/icm/cermine/cermine-impl). 


2. Run CERMINE on the set of papers and parse the token data, and convert the source CERMINE data to the csv format: 
    ```bash
    cd /datasets/s2-vl-utils
    python cermine_loader.py \
        --base_path sources \
        --cermine_path /path/to/cermine-impl-1.13-jar-with-dependencies.jar
    ```
    It will create the token table for each `sha-pid.csv` in the `sources/tokens` folder. 

## STEP2: Run visual layout detectors for getting the text block and line blocks 

```bash
python vision_model_loader.py --base_path sources
```
It will:
1. run visual layout detection for both text blocks and lines, and save them in the `<pdf-sha>-<page-id>.csv` files in the `sources/blocks` and `sources/lines` folder. 
2. combine the text block, line, and token information, create a refined version of visual layout detection, and save them in the `<pdf-sha>-<page-id>.csv` files in the `sources/condensed` folder. 

## STEP3: Assemble the annotations and export the dataset 

```bash
python condense_dataset.py \
    --annotation_folder 'sources/annotations' \
    --annotation_table 'sources/annotation_table.csv' \
    --cermine_pdf_dir 'sources/pdfs' \
    --cermine_csv_dir 'sources/tokens' \
    --vision_csv_dir 'sources/condensed' \
    --export_folder 'export/s2-vl' \  
    --config './config.json' 
```

It will convert all the source data in the source folder to a format that can be directly used for training the language models. By default, it will split the dataset into 5-folds for cross validation. The save folder will be specified in `--export_folder` configuration. There are several configurable options during the creation of the training dataset, perhaps the most important one is to specify what notion of blocks and lines to be used when constructing the dataset. Here are some available options: 

| Source of blocks | Sources of lines | Option               |
| ---------------- | ---------------- | -------------------- |
| CERMINE          | CERMINE          | - (default behavior) |
| Vision Model     | CERMINE          | `--use_vision_box`   |
| Vision Model     | Vision Model     | `--use_vision_line`  |
| Ground-Truth     | Vision Model     | `--use_gt_box`       |