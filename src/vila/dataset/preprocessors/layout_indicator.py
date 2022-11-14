from typing import List, Union, Dict, Any, Tuple
import itertools
from abc import abstractmethod

import numpy as np
import pysbd

from .base import SimplePDFDataPreprocessor, SimplePDFDataPreprocessor
from ...utils import *

Segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)


def split_token_based_on_sentences_boundary(words: List[str]) -> List[Tuple[int, int]]:
    """
    Returns: List[Tuple(int, int)]
        a list of (start, end) for token indices within each sentence
    """

    if len(words) == 0:
        return [(0, 0)]
    combined_words = " ".join(words)

    char2token_mask = np.zeros(len(combined_words), dtype=np.int)

    acc_word_len = 0
    for idx, word in enumerate(words):
        word_len = len(word) + 1
        char2token_mask[acc_word_len : acc_word_len + word_len] = idx
        acc_word_len += word_len

    segmented_sentences = Segmenter.segment(combined_words)
    sent_boundary = [(ele.start, ele.end) for ele in segmented_sentences]

    split = []
    token_id_start = 0
    for (start, end) in sent_boundary:
        token_id_end = char2token_mask[start:end].max()
        if end + 1 >= len(char2token_mask) or char2token_mask[end + 1] != token_id_end:
            token_id_end += 1  # (Including the end)
        split.append((token_id_start, token_id_end))
        token_id_start = token_id_end
    return split


class BaseLayoutIndicatorPDFDataPreprocessor(SimplePDFDataPreprocessor):
    def __init__(
        self,
        tokenizer,
        config,
        text_column_name="words",
        label_column_name="labels",
    ):

        super().__init__(tokenizer, config, text_column_name, label_column_name)

        self.added_special_separation_token = config.added_special_separation_token
        if self.added_special_separation_token == "default":
            self.added_special_separation_token = tokenizer.special_tokens_map[
                "sep_token"
            ]

    @abstractmethod
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:
        """It should be implemented differently for the functions"""

    def clean_text(self, example: Dict) -> Dict:
        """If the actual special token text, either [SEP] or [BLK], appeared in the text, 
        we won't generate a prediction for them. 
        A simple approach is just to remove the brackets [ or ], but this would 
        only work for the BERT based models. 
        """
        words = example["words"]
        for idx, word in enumerate(words):
            if word in self.special_tokens_map:
                words[idx] = words[idx].replace("[", "").replace("]", "")
        example["words"] = words

        return example

    def preprocess_sample(self, example: Dict, padding="max_length") -> Dict:

        example = self.clean_text(example)
        example, token_id_mapping_table = self.insert_layout_indicator(example)

        tokenized_inputs = self.tokenizer(
            example[self.text_column_name],
            padding=padding,
            truncation=True,
            is_split_into_words=True,
            return_overflowing_tokens=True,
        )

        # original label and bbox from the input
        labels = example[self.label_column_name]
        bboxes = example["bbox"]

        # batched labels and bboxes
        batched_labels = []
        batched_bboxes = []
        previous_word_idx = None
        encoded_word_ids = []

        for batch_id in range(len(tokenized_inputs["input_ids"])):

            word_ids = tokenized_inputs.word_ids(batch_index=batch_id)

            cur_label_ids = []
            cur_bboxes = []

            for _i, word_idx in enumerate(word_ids):

                if word_idx is None:
                    cur_label_ids.append(-100)
                    if (
                        tokenized_inputs["input_ids"][batch_id][_i]
                        == self.special_tokens_map[
                            self.tokenizer.special_tokens_map["sep_token"]
                        ]
                    ):
                        cur_bboxes.append([1000, 1000, 1000, 1000])
                    else:
                        cur_bboxes.append([0, 0, 0, 0])

                elif word_idx != previous_word_idx:
                    cur_label_ids.append(int(labels[word_idx]))
                    cur_bboxes.append(bboxes[word_idx])

                else:
                    cur_label_ids.append(
                        int(labels[word_idx]) if self.config.label_all_tokens else -100
                    )
                    cur_bboxes.append(bboxes[word_idx])

                if not (_i == 0 and word_idx is None):
                    # Only updates the word_idx after the 0th item
                    # This is important because there would be cross-batch
                    # tokens.
                    previous_word_idx = word_idx

                if word_idx is not None:
                    if tokenized_inputs["input_ids"][batch_id][_i] not in [
                        self.special_tokens_map[
                            self.tokenizer.special_tokens_map["sep_token"]
                        ],
                        self.special_tokens_map[self.added_special_separation_token],
                    ]:
                        # Because we could possibly insert [SEP] or [BLK] tokens in
                        # this process.
                        encoded_word_ids.append(word_idx)

            batched_labels.append(cur_label_ids)
            batched_bboxes.append(cur_bboxes)

            # Find the last word id in this batch to handle
            # multi-batch samples
            for word_id in reversed(word_ids):
                if word_id is not None:
                    previous_word_idx = word_id
                    break

        new_id_to_original_id = {
            ele: idx for idx, ele in enumerate(token_id_mapping_table)
        }

        tokenized_inputs["labels"] = batched_labels
        tokenized_inputs["bbox"] = batched_bboxes
        tokenized_inputs["encoded_word_ids"] = [
            new_id_to_original_id[ele] for ele in set(encoded_word_ids)
        ]

        tgt = set(range(max(tokenized_inputs["encoded_word_ids"]) + 1))
        src = set(tokenized_inputs["encoded_word_ids"])
        missing = [e for e in tgt if e not in src]
        errors = [example[self.text_column_name][token_id_mapping_table[m]] for m in missing]

        if errors:
            ord_of_errors = {ord(c) for e in errors for c in e}
            raise AssertionError(f'These char IDs get dropped in huggingface: {ord_of_errors}.\n'
                                 f'Dont forget to add: {[unicodedata.category(chr(i)) for i in ord_of_errors]}'
                                 f' categories to unicode replacement')

        return tokenized_inputs


class BlockLayoutIndicatorPDFDataPreprocessor(BaseLayoutIndicatorPDFDataPreprocessor):
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:

        processed_words = []
        processed_bbox = []
        processed_labels = []

        block_ids = example["block_ids"]
        words = example["words"]
        bbox = example["bbox"]
        labels = example["labels"]

        token_id_mapping_table = [None] * len(words)

        pre_index = 0
        new_sequence_len = 0

        for block_id, gp in itertools.groupby(block_ids):
            cur_len = len(list(gp))
            token_id_mapping_table[pre_index : pre_index + cur_len] = list(
                range(new_sequence_len, new_sequence_len + cur_len)
            )
            processed_words.extend(
                words[pre_index : pre_index + cur_len]
                + [self.added_special_separation_token]
            )
            processed_bbox.extend(
                bbox[pre_index : pre_index + cur_len]
                + [union_box(bbox[pre_index : pre_index + cur_len])]
            )
            processed_labels.extend(labels[pre_index : pre_index + cur_len] + [-100])
            pre_index += cur_len
            new_sequence_len = len(processed_labels)

        # There will be an extra [SEP] token at the end of the iterations
        processed_words = processed_words[:-1]
        processed_bbox = processed_bbox[:-1]
        processed_labels = processed_labels[:-1]

        return {
            self.text_column_name: processed_words,
            self.label_column_name: processed_labels,
            "bbox": processed_bbox,
        }, token_id_mapping_table


class RowLayoutIndicatorPDFDataPreprocessor(BaseLayoutIndicatorPDFDataPreprocessor):
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:

        processed_words = []
        processed_bbox = []
        processed_labels = []

        line_ids = example["line_ids"]  # Changed
        words = example["words"]
        bbox = example["bbox"]
        labels = example["labels"]

        token_id_mapping_table = [None] * len(words)

        pre_index = 0
        new_sequence_len = 0

        for line_id, gp in itertools.groupby(line_ids):  # Changed
            cur_len = len(list(gp))
            token_id_mapping_table[pre_index : pre_index + cur_len] = list(
                range(new_sequence_len, new_sequence_len + cur_len)
            )
            processed_words.extend(
                words[pre_index : pre_index + cur_len]
                + [self.added_special_separation_token]
            )
            processed_bbox.extend(
                bbox[pre_index : pre_index + cur_len]
                + [union_box(bbox[pre_index : pre_index + cur_len])]
            )
            processed_labels.extend(labels[pre_index : pre_index + cur_len] + [-100])
            pre_index += cur_len
            new_sequence_len = len(processed_labels)

        # There will be an extra [SEP] token at the end of the iterations
        processed_words = processed_words[:-1]
        processed_bbox = processed_bbox[:-1]
        processed_labels = processed_labels[:-1]

        return {
            self.text_column_name: processed_words,
            self.label_column_name: processed_labels,
            "bbox": processed_bbox,
        }, token_id_mapping_table


class SentenceLayoutIndicatorPDFDataPreprocessor(
    BaseLayoutIndicatorPDFDataPreprocessor
):
    def insert_layout_indicator(self, example: Dict) -> Tuple[Dict, Dict]:

        processed_words = []
        processed_bbox = []
        processed_labels = []

        words = example["words"]
        bbox = example["bbox"]
        labels = example["labels"]

        token_id_mapping_table = [None] * len(words)

        token_splits = split_token_based_on_sentences_boundary(words)

        new_sequence_len = 0
        for (start, end) in token_splits:
            token_id_mapping_table[start:end] = list(
                range(new_sequence_len, new_sequence_len + end - start)
            )
            processed_words.extend(
                words[start:end] + [self.added_special_separation_token]
            )
            processed_bbox.extend(bbox[start:end] + [union_box(bbox[start:end])])
            processed_labels.extend(labels[start:end] + [-100])

            new_sequence_len = len(processed_labels)

        # There will be an extra [SEP] token at the end of the iterations
        processed_words = processed_words[:-1]
        processed_bbox = processed_bbox[:-1]
        processed_labels = processed_labels[:-1]

        return {
            self.text_column_name: processed_words,
            self.label_column_name: processed_labels,
            "bbox": processed_bbox,
        }, token_id_mapping_table