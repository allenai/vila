import pysbd
from datasets import ClassLabel, load_dataset, load_metric

from transformers import AutoTokenizer

from vila.constants import *
from vila.dataset.preprocessors.base import SimplePDFDataPreprocessor
from vila.dataset.preprocessors.config import VILAPreprocessorConfig
from vila.dataset.preprocessors.layout_indicator import (
    BlockLayoutIndicatorPDFDataPreprocessor,
    RowLayoutIndicatorPDFDataPreprocessor,
    SentenceLayoutIndicatorPDFDataPreprocessor,
    union_box,
)
from vila.dataset.preprocessors.hierarchical_modeling import (
    RowLevelHierarchicalPDFDataPreprocessor,
    BlockLevelHierarchicalPDFDataPreprocessor,
)
from vila.dataset.preprocessors.grouping import (
    RowGroupingPDFDataPreprocessor,
    BlockGroupingPDFDataPreprocessor,
)


dummy_sample = load_dataset(
    "json", data_files="tests/fixtures/dummy_sample.json", field="data"
)["train"]

# ################################
# dummy_sample = {
#     "data": [{
#         "words":     ["Test", "the", "strange", ".", "Challenge", "Amazing"],
#         "block_ids": [0,      0,     0,         0,   0,            1],
#         "line_ids":  [0,      1,     1,         1,   2,            2],
#         "labels":    [0,      1,     1,         1,   1,            1],
#         "bbox":      [[10, 10, 20, 20],
#                       [15, 20, 25, 30],
#                       [20, 20, 30, 30],
#                       [30, 20, 31, 30],
#                       [15, 30, 25, 40],
#                       [25, 30, 35, 40]]
#     }]
# }
# ################################

tokenizer = tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True,
    use_auth_token=None,
)

config = VILAPreprocessorConfig(
    label_all_tokens = False,
    added_special_separation_token = "[SEP]",
    group_bbox_agg = "union"
)


def test_sentence_indicator_processor():

    preprocessor = SentenceLayoutIndicatorPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    Segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    combined_words = " ".join(dummy_sample["words"][0])
    encoded_sentence = "[SEP] ".join(
        ele.sent for ele in Segmenter.segment(combined_words)
    )

    assert (
        tokenizer(encoded_sentence)["input_ids"]
        == processed_version[0]["input_ids"][:9]
    )

    # fmt: off
    assert processed_version["input_ids"][0][:9] == [101, 3231, 1996, 4326, 1012, 102, 4119, 6429, 102]
    assert processed_version["labels"][0][:9] == [-100, 0, 1, 1, 1, -100, 1, 1, -100]
    assert processed_version["bbox"][0][5] == union_box(processed_version['bbox'][0][1:5])
    # fmt: on


def test_block_indicator_processor():

    preprocessor = BlockLayoutIndicatorPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    # fmt: off
    assert processed_version["input_ids"][0][:9] == [101, 3231, 1996, 4326, 1012, 4119, 102, 6429, 102]
    assert processed_version["labels"][0][:9] == [-100, 0, 1, 1, 1, 1, -100, 1, -100]
    assert processed_version["bbox"][0][6] == union_box(processed_version['bbox'][0][1:6])
    # fmt: on


def test_row_indicator_processor():
    preprocessor = RowLayoutIndicatorPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    # fmt: off
    assert processed_version["input_ids"][0][:10] == [101, 3231, 102, 1996, 4326, 1012, 102, 4119, 6429, 102]
    assert processed_version["labels"][0][:10] == [-100, 0, -100, 1, 1, 1, -100, 1, 1, -100]
    assert processed_version["bbox"][0][2] == union_box(processed_version["bbox"][0][1:2])
    assert processed_version["bbox"][0][6] == union_box(processed_version["bbox"][0][3:6])
    # fmt: on


def test_row_level_hierarchical_processor():
    preprocessor = RowLevelHierarchicalPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    # fmt: off
    assert len(processed_version["input_ids"][0]) == MAX_LINE_PER_PAGE
    assert len(processed_version["input_ids"][0][0]) == MAX_TOKENS_PER_LINE
    assert processed_version["input_ids"][0][0][:3] == [101, 3231, 102]
    assert processed_version["input_ids"][0][1][:5] == [101, 1996, 4326, 1012, 102]
    assert processed_version['input_ids'][0][2][:4] == [101, 4119, 6429, 102]

    assert processed_version['bbox'][0][0] == union_box(dummy_sample[0]['bbox'][:1])
    assert processed_version['bbox'][0][1] == union_box(dummy_sample[0]['bbox'][1:4])
    assert processed_version['bbox'][0][2] == union_box(dummy_sample[0]['bbox'][4:6])

    assert processed_version["labels"][0][:3] == [0, 1, 1]
    # fmt: on


def test_block_level_hierarchical_processor():
    preprocessor = BlockLevelHierarchicalPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    # fmt: off
    assert len(processed_version["input_ids"][0]) == MAX_BLOCK_PER_PAGE
    assert len(processed_version["input_ids"][0][0]) == MAX_TOKENS_PER_BLOCK
    assert processed_version["input_ids"][0][0][:7] == [101, 3231, 1996, 4326, 1012, 4119, 102]
    assert processed_version["input_ids"][0][1][:3] == [101, 6429, 102]

    assert processed_version['bbox'][0][0] == union_box(dummy_sample[0]['bbox'][:5])
    assert processed_version['bbox'][0][1] == union_box(dummy_sample[0]['bbox'][5:6])

    assert processed_version["labels"][0][:2] == [1, 1] # The 1st block should be 1 due to the majority voting strategy
    # fmt: on


def test_row_level_grouping_processor():
    preprocessor = RowGroupingPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    # fmt: off
    assert len(processed_version["input_ids"]) == 3
    assert processed_version["input_ids"][0] == [101, 3231, 102]
    assert processed_version["input_ids"][1] == [101, 1996, 4326, 1012, 102]
    assert processed_version['input_ids'][2] == [101, 4119, 6429, 102]

    assert processed_version['bbox'][0][1:-1] == dummy_sample[0]['bbox'][:1]
    assert processed_version['bbox'][1][1:-1] == dummy_sample[0]['bbox'][1:4]
    assert processed_version['bbox'][2][1:-1] == dummy_sample[0]['bbox'][4:6]

    assert processed_version["labels"] == [0, 1, 1]
    # fmt: on


def test_block_level_grouping_processor():
    preprocessor = BlockGroupingPDFDataPreprocessor(tokenizer, config)
    processed_version = dummy_sample.map(
        preprocessor.preprocess_batch,
        batched=True,
        remove_columns=dummy_sample.column_names,
        load_from_cache_file=False,
    )

    # fmt: off
    assert len(processed_version["input_ids"]) == 2
    assert processed_version["input_ids"][0] == [101, 3231, 1996, 4326, 1012, 4119, 102]
    assert processed_version["input_ids"][1] == [101, 6429, 102]

    assert processed_version['bbox'][0][1:-1] == dummy_sample[0]['bbox'][:5]
    assert processed_version['bbox'][1][1:-1] == dummy_sample[0]['bbox'][5:6]

    assert processed_version["labels"] == [1, 1]
    # fmt: on