# S2-VLUE 

- [S2-VLUE](#s2-vlue)
  - [Overview](#overview)
  - [Download & Usage](#download--usage)
    - [Download the exported JSON (for training language models)](#download-the-exported-json-for-training-language-models)
    - [Download the source PDFs or screenshots](#download-the-source-pdfs-or-screenshots)
  - [Datasets Details](#datasets-details)
    - [The S2-VL dataset](#the-s2-vl-dataset)
      - [Recreating the dataset from PDFs and annotations](#recreating-the-dataset-from-pdfs-and-annotations)
      - [Dataset Curation Details](#dataset-curation-details)
    - [The VILA-enhanced DocBank Dataset](#the-vila-enhanced-docbank-dataset)
  - [Dataset Details](#dataset-details)
    - [Statistics of the Datasets](#statistics-of-the-datasets)
    - [File Structures](#file-structures)
  - [Reference](#reference)
  - [Citation](#citation)

## Overview 

The S2-VLUE, Semantic Scholar **V**isual **L**ayout-enhanced Scientific Text **U**nderstanding **E**valuation (S2-VLUE) Benchmark Suite, is created to evaluate the scientific document understanding and parsing with visual layout information. 

It consists of three datasets, i.e., GROTOAP2, DocBank, and, S2-VL. We modify the existing dataset GROTOAP2[1] and DocBank[2], adding visual layout information and converting them to a format that is compatible with [HuggingFace Datasets](https://huggingface.co/docs/datasets/). 
The S2-VL dataset is a newly curated dataset that addresses three major drawbacks in existing work: 1) annotation quality, 2) VILA creation, and 3) domain coverage. 
It contains human annotations for papers from 19 scientific disciplines. 
We provide scripts for downloading the source PDF files as well as converting them to a similar HuggingFace Datasets format. 

## Download & Usage 

### Download the exported JSON (for training language models)

```bash
cd <vila-root>/datasets
bash ./download.sh <dataset-name> #grotoap2, docbank, s2-vl or all
```

### Download the source PDFs or screenshots 

- GROTOAP2 (downloading paper PDFs)
  - Please follow the instructions from the [GROTOAP2 Project README](http://cermine.ceon.pl/grotoap2/README).
- DocBank (downloading paper page screenshots)
  - Please follow the instructions from the [home page of the DocBank Project](https://doc-analysis.github.io/docbank-page/index.html). 
- S2-VL (downloading paper PDFs)
  - Please check the instructions in [s2-vl-utils/README.md](s2-vl-utils/README.md).

## Datasets Details 

### The S2-VL dataset

During the data release process, we unfortunately found that a small portion of PDFs in our dataset (22 out of 87) had additional copyright constraints of which we had been unaware. This meant that we could not directly release the data corresponding to these papers. As such, in the downloaded version, it contains only paper data created from the 65 papers. 

If you are interested in the version of the dataset used for training and evaluation in our paper, please fill out this [Google Form](https://forms.gle/M1g9tQLrUtKSsDYA7) to request access. 

#### Recreating the dataset from PDFs and annotations

We also provide the full code to help you recreate the dataset from PDFs and annotation files to the JSON files for training models. Please check the instructions in [s2-vl-utils/README.md](s2-vl-utils/README.md).

#### Dataset Curation Details 

Please find a detailed description of the labeling schemas and categories in the following documents:
- [Labeling Instruction](https://docs.google.com/document/d/1DsIDKNEi8GBxrqQuYRy86lCKhksgvyRaGhXPCheGgG0/edit?usp=sharing)
- [S2-VL Category Definition](https://docs.google.com/document/d/1frGmzYOHnVRWAwTOuuPfc3KVAwu-XKdkFSbpLfy78RI/edit?usp=sharing)
  - We labeled both layout and semantic categories in S2-VL (see the document above), but only the 15* layout categories will be used in this evaluation benchmark. 
- [The 19 Scientific Disciplines](https://docs.google.com/document/d/1ytJkYhswp4Wlx8tT1iRe-jdjx5A-nqisvUikgmqSQKc/edit?usp=sharing)

*The `algorithm` category is removed due to its small number of instances. 

### The VILA-enhanced DocBank Dataset

## Dataset Details 

### Statistics of the Datasets

|                   | GROTOAP2     | DocBank         | S2-VL-ver1                       |
| ----------------- | ------------ | --------------- | ------------------------------ |
| Train Test Split  | 83k/18k/18k  | 398k/50k/50k    | *                              |
| Annotation Method | Automatic    | Automatic       | Human Annotation               |
| Paper Domain      | Life Science | Math/Physics/CS | 19 Disciplines                 |
| VILA Structure    | PDF parsing  | Vision model    | Gold Label / Detection methods |
| # of Categories   | 22           | 12              | 15                             |

|                           | GROTOAP2 | DocBank | S2-VL-ver1* |
| ------------------------- | -------- | ------- | --------- |
| **Tokens per Page**       |
| Average                   | 1203     | 838     | 790       |
| Std                       | 591      | 503     | 453       |
| 95th Percentile           | 2307     | 1553    | 1591      |
| **Text Lines per Page**   |
| Average                   | 90       | 60      | 64        |
| Std                       | 51       | 34      | 54        |
| 95th Percentile           | 171      | 125     | 154       |
| **Text Blocks per Page**  |
| Average                   | 12       | 15      | 22        |
| Std                       | 16       | 8       | 36        |
| 95th Percentile           | 37       | 30      | 68        |
| **Tokens per Text Line**  |
| Average                   | 17       | 16      | 14        |
| Std                       | 12       | 43      | 10        |
| 95th Percentile           | 38       | 38      | 30        |
| **Tokens per Text Block** |
| Average                   | 90       | 57      | 48        |
| Std                       | 184      | 138     | 121       |
| 95th Percentile           | 431      | 210     | 249       |

* This is calculated based on the S2-VL-ver1 with all 87 papers.

### File Structures 

1. The organization of the dataset files :
    ```bash
    grotoap2 # Docbank is similar 
    ├─ labels.json       
    ├─ train-token.json
    ├─ dev-token.json           
    ├─ test-token.json           
    └─ train-test-split.json
    ```
2. What's in each file?
    1. `labels.json`
        ```json
        {"0": "Title",
         "1": "Author",
         ...
        }
        ```
    2. `train-test-split.json`
        ```json
        {
            "train": [
                "pdf-file-name", ...
            ],
            "test": ["pdf-file-name", ...]
        }
        ```
    3. `train-token.json`, `dev-token.json` or `test-token.json`
        Please see detailed schema explanation in the [schema-token.json](schema-token.json) file.
3. Special notes on the folder structure for S2-VL: since the dataset size is small, we use 5-fold cross validation in the paper. The released version has a similar structure: 
    ```bash
    s2-vl-ver1
    ├─ 0  # 5-fold Cross validation                           
    │  ├─ labels.json               
    │  ├─ test-token.json           
    │  ├─ train-test-split.json     
    │  └─ train-token.json          
    ├─ 1  # fold-1, have the same files as other folds                         
    │  ├─ labels.json               
    │  ├─ test-token.json           
    │  ├─ train-test-split.json     
    │  └─ train-token.json          
    ├─ 2                            
    ├─ 3                            
    └─ 4
    ```

## Reference 

1. The GROTOAP2 Dataset: 
    - Paper: https://www.dlib.org/dlib/november14/tkaczyk/11tkaczyk.html
    - Original download link: http://cermine.ceon.pl/grotoap2/
    - Licence: Open Access license

2. The Original DocBank Dataset: 
    - Paper: https://arxiv.org/pdf/2006.01038.pdf
    - Original download link: https://github.com/doc-analysis/DocBank
    - Licence: Apache-2.0

## Citation 

```
@article{Shen2021IncorporatingVL,
  title={Incorporating Visual Layout Structures for Scientific Text Classification},
  author={Zejiang Shen and Kyle Lo and Lucy Lu Wang and Bailey Kuehl and Daniel S. Weld and Doug Downey},
  journal={ArXiv},
  year={2021},
  volume={abs/2106.00676},
  url={https://arxiv.org/abs/2106.00676}
}
```
