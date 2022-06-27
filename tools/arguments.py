from dataclasses import dataclass, field
from typing import Optional

import sys

sys.path.append("../src")
from vila.constants import *


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    #################################
    ######### VILA Settings #########
    #################################

    added_special_separation_token: str = field(
        default="SEP",
        metadata={
            "help": "The added special token for I-VILA models for separating the blocks/sentences/rows. Can be one of {SEP, BLK}. Default to `SEP`."
        },
    )
    textline_encoder_output: str = field(
        default="cls",
        metadata={
            "help": "How to obtain the group representation from the H-VILA model? Can be one of {cls, sep, average, last}. Default to `cls`."
        },
    )
    not_resume_training: bool = field(
        default=False,
        metadata={"help": "whether resume training from the existing checkpoints."},
    )

    def __post_init__(self):

        assert self.added_special_separation_token in ["BLK", "SEP"]

        if self.added_special_separation_token == "BLK":
            self.added_special_separation_token = "[BLK]"

        if self.added_special_separation_token == "SEP":
            self.added_special_separation_token = "[SEP]"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    label_map_file: Optional[str] = field(
        default=None,
        metadata={"help": "The JSON file storing the (id:label_name)"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    #################################
    ######### VILA Settings #########
    #################################

    agg_level: str = field(
        default="block",
        metadata={
            "help": "Used in some scenarios where the models will inject additional information to the models based on the agg_level"
        },
    )
    group_bbox_agg: str = field(
        default="first",
        metadata={
            "help": "The method to get the group bounding bbox, one of {union, first, center, last}. Default to `first`."
        },
    )
    max_line_per_page: Optional[int] = field(
        default=None,
        metadata={"help": "The number of textlines per page"},
    )
    max_tokens_per_line: Optional[int] = field(
        default=None,
        metadata={"help": "The number of tokens per textline"},
    )
    max_block_per_page: Optional[int] = field(
        default=None,
        metadata={"help": "The number of block per page"},
    )
    max_tokens_per_block: Optional[int] = field(
        default=None,
        metadata={"help": "The number of tokens per block"},
    )
    pred_file_name: Optional[str] = field(
        default="test_predictions.csv",
        metadata={"help": "The filename used for saving predictions."},
    )
    test_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use for prediction."},
    )

    def __post_init__(self):

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."

        self.task_name = self.task_name.lower()

        if self.dataset_name is not None:
            if self.dataset_name.lower() == "grotoap2":
                # fmt:off
                self.train_file      = "../data/grotoap2/train-token.json"
                self.validation_file = "../data/grotoap2/dev-token.json"
                self.test_file       = "../data/grotoap2/test-token.json"
                self.label_map_file  = "../data/grotoap2/labels.json"
                self.dataset_name = None
                # fmt:on

                self.max_line_per_page = MAX_LINE_PER_PAGE
                self.max_tokens_per_line = MAX_TOKENS_PER_LINE
                self.max_block_per_page = MAX_BLOCK_PER_PAGE
                self.max_tokens_per_block = MAX_TOKENS_PER_BLOCK

            elif self.dataset_name.lower() == "docbank":
                
                # fmt:off
                self.train_file      = "../data/docbank/train-token.json"
                self.validation_file = "../data/docbank/dev-token.json"
                self.test_file       = "../data/docbank/test-token.json"
                self.label_map_file  = "../data/docbank/labels.json"
                self.dataset_name    = None
                # fmt:on

                self.max_line_per_page = 100
                self.max_tokens_per_line = 25
                self.max_block_per_page = 30
                self.max_tokens_per_block = 100