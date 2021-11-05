# Recreating the S2-VL Dataset 

- [Recreating the S2-VL Dataset](#recreating-the-s2-vl-dataset)
  - [STEP0: Install extra dependencies for creating the dataset](#step0-install-extra-dependencies-for-creating-the-dataset)
  - [STEP1: Download the papers](#step1-download-the-papers)
  - [STEP2: Parse token data using CERMINE](#step2-parse-token-data-using-cermine)
  - [STEP3: Run visual layout detectors for getting the text block and line blocks](#step3-run-visual-layout-detectors-for-getting-the-text-block-and-line-blocks)
  - [STEP4: Assemble the annotations and export the dataset](#step4-assemble-the-annotations-and-export-the-dataset)

## STEP0: Install extra dependencies for creating the dataset 

```bash
cd /datasets/s2-vl-utils
# activate the corresponding environment 
pip install -r requirements
```

## STEP1: Download the papers 

```bash
python download.py --base-path sources/s2-vl-ver1
```
This will download the pdf files to `sources/s2-vl-ver1/pdfs`. 
We'll check and report PDFs that don't have the compatible SHA1 code or cannot be downloaded. 
Note: when you find incompatible SHAs for one PDF, it doesn't necessarily mean the PDFs are different. 

## STEP2: Parse token data using CERMINE 

1. Download JAVA and CERMINE following instructions in [this repo](https://github.com/CeON/CERMINE#using-cermine) (PS: The easiest approach would be just downloading CERMINE v1.13 from [JFrog](http://maven.ceon.pl/artifactory/webapp/#/artifacts/browse/simple/General/kdd-releases/pl/edu/icm/cermine/cermine-impl). 


2. Run CERMINE on the set of papers and parse the token data, and convert the source CERMINE data to the csv format: 
    ```bash
    python cermine_loader.py \
        --base-path sources/s2-vl-ver1 \
        --cermine-path /path/to/cermine-impl-1.13-jar-with-dependencies.jar
    ```
    It will create the token table for each `sha-pid.csv` in the `sources/tokens` folder. 

## STEP3: Run visual layout detectors for getting the text block and line blocks 

```bash
python vision_model_loader.py --base-path sources
```
It will:
1. run visual layout detection for both text blocks and lines, and save them in the `<pdf-sha>-<page-id>.csv` files in the `sources/blocks` and `sources/lines` folder. 
2. combine the text block, line, and token information, create a refined version of visual layout detection, and save them in the `<pdf-sha>-<page-id>.csv` files in the `sources/condensed` folder. 

## STEP4: Assemble the annotations and export the dataset 

```bash
python condense_dataset.py \
    --annotation-folder 'sources/s2-vl-ver1/annotations' \
    --annotation-table 'sources/s2-vl-ver1/annotation_table.csv' \
    --cermine-pdf-dir 'sources/s2-vl-ver1/pdfs' \
    --cermine-csv-dir 'sources/s2-vl-ver1/tokens' \
    --vision-csv-dir 'sources/s2-vl-ver1/condensed' \
    --export-folder 'export/s2-vl-ver1' \  
    --config './config.json' 
```

It will convert all the source data in the source folder to a format that can be directly used for training the language models. By default, it will split the dataset into 5-folds for cross validation. The save folder will be specified in `--export-folder` configuration. There are several configurable options during the creation of the training dataset, perhaps the most important one is to specify what notion of blocks and lines to be used when constructing the dataset. Here are some available options: 

| Source of blocks | Sources of lines | Option               |
| ---------------- | ---------------- | -------------------- |
| CERMINE          | CERMINE          | - (default behavior) |
| Vision Model     | CERMINE          | `--use-vision-box`   |
| Vision Model     | Vision Model     | `--use-vision-line`  |
| Ground-Truth     | Vision Model     | `--use-gt-box`       |