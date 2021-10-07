# S2-VLUE 

The S2-VLUE, Semantic Scholar **V**isual **L**ayout-enhanced Scientific Text **U**nderstanding **E**valuation (S2-VLUE) Benchmark Suite, is created to evaluate the scientific document understanding and parsing with visual layout information. 

It consists of three datasets, i.e., GROTOAP2, DocBank, and, S2-VL. We modify the existing dataset GROTOAP2[1] and DocBank[2], adding visual layout information and converting them to a format that is compatible with [HuggingFace Datasets](https://huggingface.co/docs/datasets/). 
The S2-VL dataset is a newly curated dataset that addresses three major drawbacks in existing work: 1) annotation quality, 2) VILA creation, and 3) domain coverage. 
It contains human annotations for papers from 19 scientific disciplines. 
We will provide scripts for downloading the source PDF files as well as converting them to a similar HuggingFace Datasets format. 

## Download & Usage 

### Download to folders

```bash
cd <vila-root>/datasets
bash ./download.sh <dataset-name> #grotoap2, docbank, s2-vl or all
```

### Directly loading from HuggingFace Datasets
WIP

## Datasets Statistics

### Overall 
|                   | GROTOAP2     | DocBank         | S2-VL-v1                       |
| ----------------- | ------------ | --------------- | ------------------------------ |
| Train Test Split  | 83k/18k/18k  | 398k/50k/50k    | *                              |
| Annotation Method | Automatic    | Automatic       | Human Annotation               |
| Paper Domain      | Life Science | Math/Physics/CS | 19 Domains                     |
| VILA Structure    | PDF parsing  | Vision model    | Gold Label / Detection methods |
| # of Categories   | 22           | 12              | 15                             |

### Document Details 

|                           | GROTOAP2 | DocBank | S2-VL-v1* |
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

* This is calculated based on "old" data and should be updated. 

## Details about the VILA-enhanced DocBank dataset

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
