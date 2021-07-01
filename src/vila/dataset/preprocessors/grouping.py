from typing import List, Union, Dict, Any, Tuple
import itertools
import math
from abc import abstractmethod
from collections import Counter

from .base import SimplePDFDataPreprocessor
from ...utils import *


def clean_group_ids(lst):
    _lst = []
    for idx, (_, gp) in enumerate(itertools.groupby(lst)):
        _lst.extend([idx] * len(list(gp)))
    return _lst


def split_example_based_on(example: Dict, level: str) -> List[Dict]:

    level_target = example[f"{level}_ids"]
    level_target = clean_group_ids(level_target)
    keys = list(example.keys())

    pre_index = 0
    regrouped_sequence = {key: [] for key in keys}

    num_tokens_per_group = {}

    for level_idx, (orig_level_id, gp) in enumerate(itertools.groupby(level_target)):
        cur_len = len(list(gp))

        for key in keys:
            regrouped_sequence[key].append(
                example[key][pre_index : pre_index + cur_len]
            )

        pre_index += cur_len
        num_tokens_per_group[orig_level_id] = cur_len

    return regrouped_sequence, num_tokens_per_group


class BaseGroupingPDFDataPreprocessor(SimplePDFDataPreprocessor):
    @abstractmethod
    def preprocess_sequence(self, example: Dict) -> Tuple[Dict, Dict]:
        """It should be implemented differently for the functions"""

    def preprocess_sample(self, example: Dict, padding="max_length") -> Dict:

        splitted_example, num_tokens_per_group = self.preprocess_sequence(example)

        tokenized_inputs = self.tokenizer(
            splitted_example[self.text_column_name],
            padding=padding,
            truncation=True,
            is_split_into_words=True,
        )

        # original label and bbox from the input
        labels = []
        bboxes = []

        for i, (label, bbox) in enumerate(
            zip(splitted_example[self.label_column_name], splitted_example["bbox"])
        ):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            cur_bboxes = []
            for _i, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    if (
                        tokenized_inputs["input_ids"][i][_i]
                        == self.special_tokens_map[
                            self.tokenizer.special_tokens_map["pad_token"]
                        ]
                    ):
                        cur_bboxes.append([1000, 1000, 1000, 1000])
                        # A lazy solution for now as only [SEP] token
                        # has the [1000, 1000, 1000, 1000] bbox

                    else:
                        cur_bboxes.append([0, 0, 0, 0])

                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(int(label[word_idx]))
                    cur_bboxes.append(bbox[word_idx])

                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        int(label[word_idx]) if self.config.label_all_tokens else -100
                    )
                    cur_bboxes.append(bbox[word_idx])

                previous_word_idx = word_idx

            if len(label_ids) == 0:
                label_id = -100
            else:
                label_freq = Counter(label_ids)
                if len(label_freq) == 1 and -100 in label_freq:
                    label_id = -100
                else:
                    label_freq.pop(-100)
                    label_id = label_freq.most_common(1)[0][0]

            labels.append(label_id)
            bboxes.append(cur_bboxes)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["num_tokens_per_group"] = list(num_tokens_per_group.items())
        return tokenized_inputs

    def preprocess_batch(self, examples: Dict[str, List]) -> Dict[str, List]:
        """This is a wrapper based on the preprocess_sample. There might be
        considerable performance loss, but we don't need to rewrite the main
        iteration loop again.
        """
        all_processed_examples = []
        for example in self.iter_example(examples):
            processed_example = self.preprocess_sample(example, padding=False)
            processed_example.pop("num_tokens_per_group")
            # encoded_word_ids will only be used in test time
            all_processed_examples.append(processed_example)

        return self.batchsize_examples(all_processed_examples)


class RowGroupingPDFDataPreprocessor(BaseGroupingPDFDataPreprocessor):
    def preprocess_sequence(self, example: Dict) -> Tuple[Dict, Dict]:
        return split_example_based_on(example, "line")


class BlockGroupingPDFDataPreprocessor(BaseGroupingPDFDataPreprocessor):
    def preprocess_sequence(self, example: Dict) -> Tuple[Dict, Dict]:
        return split_example_based_on(example, "block")
