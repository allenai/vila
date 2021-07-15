# VILA🌴 <br> Incorporating **VI**sual **LA**yout Structures for Scientific Text Classification

<a href="https://arxiv.org/abs/2106.00676"><img src="https://img.shields.io/badge/arXiv-2106.00676-b31b1b.svg" title="Layout Parser Paper"></a>
<a href="https://github.com/Layout-Parser/layout-parser/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/layoutparser" title="Layout Parser uses Apache 2 License"></a>

## Motivation 

Scientific papers typically organize contents in visual groups like text blocks or lines, and text within each group usually have the same semantics. We explore different approaches for injecting the group structure into the text classifiers, and build models that improves the accuracy or efficiency of the scientific text classification task. 

![tease](./.github/github-teaser.png)

## Installation 

After cloning the github repo, you can either install the `vila` library or just install the dependencies: 
```bash 
git clone git@github.com:allenai/VILA.git
cd VILA 
conda create -n vila python=3.6
pip install -e . # Install the `vila` library 
pip install -r requirements.txt # Only install the dependencies 
```

We tested the code and trained the models using `Python≥3.6`, `PyTorch==1.7.1`, and `transformers==4.4.2`. 

## Usage 

### Directory Structure 

```
VILA
├─ checkpoints  # For all trained weights 
│  └─ grotoap2  # For each dataset                                 
│     ├─ baseline  # For the experiment type, e.g., baseline, ivila, hvila, ...
│     │  └─ bert-base-uncased  # For the used base model, e.g., bert-base-uncased. 
│     │     ├─ checkpoint-199999                                
│     │     ├─ checkpoint-299999                                 
│     │     ├─ all_results.json                                       
│     │     └─ pytorch_model.bin                         
│     └─ ivila-BLK-row                           
│        └─ microsoft-layoutlm-base-uncased 
└─ data                                       
   ├─ docbank
   ├─ ...
   └─ grotoap2                                 

```

Note:
- We will provide the download links to the datasets very soon. 

### Training 

All training scripts are in the `./scripts` folder. 

1. Train the baseline models
    ```bash
    cd scripts
    # bash train_baseline.sh [dataset-name] [base-model-name]
    bash train_baseline.sh grotoap2 bert-base-uncased
    bash train_baseline.sh docbank microsoft/layoutlm-base-uncased
    ```
2. Train the I-VILA models
    ```bash
    cd scripts
    # bash train_ivila.sh [dataset-name] [how-to-obtain-layout-indicators] [used-special-token] [base-model-name]
    bash train_ivila.sh grotoap2 row BLK microsoft/layoutlm-base-uncased 
      # Row is an alias for textline 
    bash train_ivila.sh docbank block SEP bert-base-uncased
      # We can also use the default special tokens like SEP 
    bash train_ivila.sh s2-vl sentence BLK roberta-base 
      # We can also extract the sentence breaks using spacy and use them as indicators.
    ```

### Model Inference/Prediction 

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

