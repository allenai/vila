# S2-VLUE 

The S2-VLUE, Semantic Scholar **V**isual **L**ayout-enhanced Scientific Text **U**nderstanding **E**valuation (S2-VLUE) Benchmark Suite, is created to evaluate the scientific document understanding and parsing with visual layout information. 

It consists of three datasets, i.e., GROTOAP2, DocBank, and, S2-VL. We modify the existing dataset GROTOAP2[1] and DocBank[2], adding visual layout information and converting them to a format that is compatible with [HuggingFace Datasets](https://huggingface.co/docs/datasets/). 
The S2-VL dataset is a newly curated dataset that addresses three major drawbacks in existing work: 1) annotation quality, 2) VILA creation, and 3) domain coverage. 
It contains human annotations for papers from 19 scientific disciplines. 
We will provide scripts for downloading the source PDF files as well as converting them to a similar HuggingFace Datasets format. 

## Download & Usage 

## Datasets Statistics

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
