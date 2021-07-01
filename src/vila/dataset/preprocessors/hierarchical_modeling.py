import itertools
import math
from collections import Counter

from ...constants import *
from .base import BasePDFDataPreprocessor
from ...utils import *


def find_idx_in_list(lst, target_value, start=0):
    """Find the max idx in the list such that lst[idx] < target_value.
    If list is empty or the first value is larger than the target value, return -1
    """
    if len(lst) == 0 or lst[0] > target_value:
        return -1
    for idx in range(start, len(lst)):
        if lst[idx] >= target_value:
            return idx - 1
    return idx + 1


def get_most_common_element(lst):
    return Counter(lst).most_common(1)[0][0]


def remap_group_id(lst):
    reindex_element = {val: idx for idx, val in enumerate(set(lst))}
    return [reindex_element[ele] for ele in lst]


def clean_group_ids(lst):
    _lst = []
    for idx, (_, gp) in enumerate(itertools.groupby(lst)):
        _lst.extend([idx] * len(list(gp)))
    return _lst


class BaseHierarchicalPDFDataPreprocessor(BasePDFDataPreprocessor):

    GROUP_BBOX_AGG_FUNC = {
        "union": union_box,
        "first": lambda boxes: boxes[0] if len(boxes) > 0 else [0, 0, 0, 0],
        "last": lambda boxes: boxes[-1] if len(boxes) > 0 else [0, 0, 0, 0],
        "center": lambda boxes: boxes[len(boxes) // 2]
        if len(boxes) > 0
        else [0, 0, 0, 0],
    }

    def __init__(self, tokenizer, config):

        self.tokenizer = tokenizer
        self.config = config
        self.group_bbox_agg_func = self.GROUP_BBOX_AGG_FUNC[config.group_bbox_agg]

        self.max_line_per_page = getattr(config, "max_line_per_page", MAX_LINE_PER_PAGE)
        self.max_tokens_per_line = getattr(
            config, "max_tokens_per_line", MAX_TOKENS_PER_LINE
        )
        self.max_block_per_page = getattr(
            config, "max_block_per_page", MAX_BLOCK_PER_PAGE
        )
        self.max_tokens_per_block = getattr(
            config, "max_tokens_per_block", MAX_TOKENS_PER_BLOCK
        )

    def preprocess_batch(self, examples):

        all_processed_examples = []
        for example in self.iter_example(examples):
            processed_example = self.preprocess_sample(example)
            # encoded_word_ids will only be used in test time
            all_processed_examples.append(processed_example)

        return self.batchsize_examples(all_processed_examples)


class RowLevelHierarchicalPDFDataPreprocessor(BaseHierarchicalPDFDataPreprocessor):
    def preprocess_chunked_sample(self, examples):

        processed_batches = []

        max_textline_len = float("-inf")
        for line_ids, words, bbox, labels in zip(
            examples["line_ids"],
            examples["words"],
            examples["bbox"],
            examples["labels"],
        ):
            line_words = []
            line_bbox = []
            line_labels = []
            line_word_cnt = {}  # Dict is ordered since Python 3.5

            pre_index = 0

            for line_id, (_orig_line_id, gp) in enumerate(itertools.groupby(line_ids)):
                if line_id >= self.max_line_per_page:
                    line_id -= 1  # Recover the correct line_id for generating the line_attention_mask
                    break
                cur_len = len(list(gp))
                line_word_cnt[_orig_line_id] = cur_len
                line_words.append(words[pre_index : pre_index + cur_len])
                line_bbox.append(bbox[pre_index : pre_index + cur_len])
                line_labels.append(
                    get_most_common_element(labels[pre_index : pre_index + cur_len])
                )  # Because all tokens in the same line have the same labels
                pre_index += cur_len
                max_textline_len = max(max_textline_len, cur_len)

            line_attention_mask = [1] * (line_id + 1) + [0] * (
                self.max_line_per_page - line_id - 1
            )

            if line_id < self.max_line_per_page:
                line_words.extend(
                    [[self.tokenizer.special_tokens_map["pad_token"]]]
                    * (self.max_line_per_page - line_id - 1)
                )
                line_bbox.extend(
                    [[[0, 0, 0, 0]]] * (self.max_line_per_page - line_id - 1)
                )
                line_labels.extend([-100] * (self.max_line_per_page - line_id - 1))

            tokenized_line = self.tokenizer(
                line_words,
                padding="max_length",
                max_length=self.max_tokens_per_line,
                truncation=True,
                is_split_into_words=True,
            )

            tokenized_line["bbox"] = [
                self.group_bbox_agg_func(bboxes) for bboxes in line_bbox
            ]
            tokenized_line["labels"] = line_labels
            tokenized_line["group_level_attention_mask"] = line_attention_mask
            tokenized_line["group_word_count"] = list(line_word_cnt.items())

            processed_batches.append(tokenized_line)

        condensed_batch = {
            key: [ele[key] for ele in processed_batches]
            for key in processed_batches[0].keys()
        }

        del processed_batches
        return condensed_batch

    def preprocess_sample(self, example):

        if False:
            # Keep for future reference
            line_count_this_page = len(set(example["line_ids"]))
            max_line_id = max(example["line_ids"])

            if max_line_id != line_count_this_page:
                line_ids = remap_group_id(example["line_ids"])
            else:
                line_ids = example["line_ids"]
        else:
            line_ids = clean_group_ids(example["line_ids"])
            line_count_this_page = max(line_ids) + 1

        if line_count_this_page <= self.max_line_per_page:
            # In this case, we just need to send it to the regular tokenizer
            batched_input = self.preprocess_chunked_sample(
                {
                    "words": [example["words"]],
                    "labels": [example["labels"]],
                    "bbox": [example["bbox"]],
                    "line_ids": [line_ids],
                }
            )
        else:
            # We need to split the input and regroup it into multiple batches
            num_splits = math.ceil(line_count_this_page / self.max_line_per_page)
            assert num_splits > 1

            words = example["words"]
            labels = example["labels"]
            bbox = example["bbox"]

            newly_batched_words = []
            newly_batched_labels = []
            newly_batched_bbox = []
            newly_batched_line_ids = []

            prev_idx = 0
            for line_id_split in range(
                self.max_line_per_page,
                (num_splits + 1) * self.max_line_per_page,
                self.max_line_per_page,
            ):

                cur_idx = find_idx_in_list(line_ids, line_id_split, prev_idx)
                assert cur_idx > prev_idx  # Ensure the new segment is not empty

                newly_batched_words.append(words[prev_idx : cur_idx + 1])
                newly_batched_labels.append(labels[prev_idx : cur_idx + 1])
                newly_batched_bbox.append(bbox[prev_idx : cur_idx + 1])
                newly_batched_line_ids.append(line_ids[prev_idx : cur_idx + 1])
                prev_idx = cur_idx + 1

            new_examples = {
                "words": newly_batched_words,
                "labels": newly_batched_labels,
                "bbox": newly_batched_bbox,
                "line_ids": newly_batched_line_ids,
            }
            batched_input = self.preprocess_chunked_sample(new_examples)

        group_word_count = [
            ele[1] for batch in batched_input["group_word_count"] for ele in batch
        ]
        assert sum(group_word_count) == len(example["words"])
        return batched_input


class BlockLevelHierarchicalPDFDataPreprocessor(BaseHierarchicalPDFDataPreprocessor):
    def preprocess_chunked_sample(self, examples):

        processed_batches = []

        max_textblock_len = float("-inf")
        for block_ids, words, bbox, labels in zip(
            examples["block_ids"],
            examples["words"],
            examples["bbox"],
            examples["labels"],
        ):
            block_words = []
            block_bbox = []
            block_labels = []
            block_word_cnt = {}  # Dict is ordered since Python 3.5

            pre_index = 0

            for block_id, (_orig_block_id, gp) in enumerate(
                itertools.groupby(block_ids)
            ):
                if block_id >= self.max_block_per_page:
                    block_id -= 1  # Recover the correct block_id for generating the block_attention_mask
                    break
                cur_len = len(list(gp))
                block_word_cnt[_orig_block_id] = cur_len
                block_words.append(words[pre_index : pre_index + cur_len])
                block_bbox.append(bbox[pre_index : pre_index + cur_len])
                block_labels.append(
                    get_most_common_element(labels[pre_index : pre_index + cur_len])
                )  # Because all tokens in the same block have the same labels
                pre_index += cur_len
                max_textblock_len = max(max_textblock_len, cur_len)

            block_attention_mask = [1] * (block_id + 1) + [0] * (
                self.max_block_per_page - block_id - 1
            )

            if block_id < self.max_block_per_page:
                block_words.extend(
                    [[self.tokenizer.special_tokens_map["pad_token"]]]
                    * (self.max_block_per_page - block_id - 1)
                )
                block_bbox.extend(
                    [[[0, 0, 0, 0]]] * (self.max_block_per_page - block_id - 1)
                )
                block_labels.extend([-100] * (self.max_block_per_page - block_id - 1))

            tokenized_block = self.tokenizer(
                block_words,
                padding="max_length",
                max_length=self.max_tokens_per_block,
                truncation=True,
                is_split_into_words=True,
            )

            tokenized_block["bbox"] = [
                self.group_bbox_agg_func(bboxes) for bboxes in block_bbox
            ]
            tokenized_block["labels"] = block_labels
            tokenized_block["group_level_attention_mask"] = block_attention_mask
            tokenized_block["group_word_count"] = list(block_word_cnt.items())

            processed_batches.append(tokenized_block)

        condensed_batch = {
            key: [ele[key] for ele in processed_batches]
            for key in processed_batches[0].keys()
        }

        del processed_batches
        return condensed_batch

    def preprocess_sample(self, example):

        if False:
            # Keep for future reference
            block_count_this_page = len(set(example["block_ids"]))
            max_block_id = max(example["block_ids"])

            if max_block_id != block_count_this_page:
                block_ids = remap_group_id(example["block_ids"])
            else:
                block_ids = example["block_ids"]
        else:
            block_ids = clean_group_ids(example["block_ids"])
            block_count_this_page = max(block_ids) + 1

        if block_count_this_page <= self.max_block_per_page:
            # In this case, we just need to send it to the regular tokenizer
            batched_input = self.preprocess_chunked_sample(
                {
                    "words": [example["words"]],
                    "labels": [example["labels"]],
                    "bbox": [example["bbox"]],
                    "block_ids": [block_ids],
                }
            )
        else:
            # We need to split the input and regroup it into multiple batches
            num_splits = math.ceil(block_count_this_page / self.max_block_per_page)
            assert num_splits > 1

            words = example["words"]
            labels = example["labels"]
            bbox = example["bbox"]

            newly_batched_words = []
            newly_batched_labels = []
            newly_batched_bbox = []
            newly_batched_block_ids = []

            prev_idx = 0
            for block_id_split in range(
                self.max_block_per_page,
                (num_splits + 1) * self.max_block_per_page,
                self.max_block_per_page,
            ):

                cur_idx = find_idx_in_list(block_ids, block_id_split, prev_idx)
                assert cur_idx > prev_idx  # Ensure the new segment is not empty

                newly_batched_words.append(words[prev_idx : cur_idx + 1])
                newly_batched_labels.append(labels[prev_idx : cur_idx + 1])
                newly_batched_bbox.append(bbox[prev_idx : cur_idx + 1])
                newly_batched_block_ids.append(block_ids[prev_idx : cur_idx + 1])
                prev_idx = cur_idx + 1

            new_examples = {
                "words": newly_batched_words,
                "labels": newly_batched_labels,
                "bbox": newly_batched_bbox,
                "block_ids": newly_batched_block_ids,
            }
            batched_input = self.preprocess_chunked_sample(new_examples)

        group_word_count = [
            ele[1] for batch in batched_input["group_word_count"] for ele in batch
        ]
        assert sum(group_word_count) == len(example["words"])
        return batched_input