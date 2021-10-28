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
python cermine_loader.py \
    --cermine_path /path/to/cermine-impl-1.13-jar-with-dependencies.jar
```

## STEP2: Run visual layout detectors for getting the text block and line blocks 

```bash
python vision_model_loader.py
```

## STEP3: Assemble the annotations and export the dataset 
