from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import json
from dataclasses import dataclass
import inspect
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def write_json(data, filename):
    with open(filename, "w") as fp:
        json.dump(data, fp)


def union_box(blocks):
    if len(blocks) == 0:
        logging.debug("The length of blocks is 0!")
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")
    for bbox in blocks:
        x1 = min(x1, bbox[0])
        y1 = min(y1, bbox[1])
        x2 = max(x2, bbox[2])
        y2 = max(y2, bbox[3])
    return [int(x1), int(y1), int(x2), int(y2)]


def columns_used_in_model_inputs(dataset, model):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    columns = [k for k in signature_columns if k in dataset.column_names]
    return columns


def classification_report(y_true, y_pred) -> Dict[str, Any]:

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean()
    keys = np.unique(y_true)
    scores = precision_recall_fscore_support(
        y_true, y_pred, labels=list(keys), zero_division=0
    )
    df = pd.DataFrame(
        scores, columns=keys, index=["precision", "recall", "f-score", "support"]
    )
    marco_scores = df.mean(axis=1)

    return {
        "accuracy": accuracy,
        "precision": marco_scores["precision"],
        "recall": marco_scores["recall"],
        "fscore": marco_scores["f-score"],
        "detailed": df.to_dict(),
    }


@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [
                label + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + label
                for label in labels
            ]

        if "bbox" in features[0]:
            if padding_side == "right":
                batch["bbox"] = [
                    f["bbox"] + [[0, 0, 0, 0]] * (sequence_length - len(f["bbox"]))
                    for f in features
                ]
            else:
                batch["bbox"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(f["bbox"])) + f["bbox"]
                    for f in features
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


@dataclass
class DataCollatorForSequenceClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        if "bbox" in features[0]:
            if padding_side == "right":
                batch["bbox"] = [
                    f["bbox"] + [[0, 0, 0, 0]] * (sequence_length - len(f["bbox"]))
                    for f in features
                ]
            else:
                batch["bbox"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(f["bbox"])) + f["bbox"]
                    for f in features
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch