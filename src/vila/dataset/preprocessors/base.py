from typing import List, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod
import itertools


class BasePDFDataPreprocessor:
    """Preprocess the raw data in PDFData format to supply
    for prediction models.
    """

    def __init__(self, tokenizer, config):

        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def preprocess_sample(self, example: Dict) -> Dict:
        """Process one example in the dataset. This function will be used
        in test time.
        """

    @abstractmethod
    def preprocess_batch(self, example: Dict[str, List]) -> Dict[str, List]:
        """Convert a batch of examples in the dataset. This function will
        be used for train and eval.
        """

    @staticmethod
    def iter_example(examples: Dict[str, List]):
        keys = list(examples.keys())
        bz = len(examples[keys[0]])
        for batch_id in range(bz):
            yield {key: examples[key][batch_id] for key in keys}

    @staticmethod
    def batchsize_examples(examples: List[Dict[str, List]]) -> Dict[str, List]:
        keys = list(examples[0].keys())
        return {
            key: list(
                itertools.chain.from_iterable(example[key] for example in examples)
            )
            for key in keys
        }


class SimplePDFDataPreprocessor(BasePDFDataPreprocessor):
    def __init__(
        self,
        tokenizer,
        config,
        text_column_name="words",
        label_column_name="labels",
    ):

        self.tokenizer = tokenizer
        self.config = config
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

        self.special_tokens_map = {
            tok: idx
            for tok, idx in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)
        }

    def preprocess_sample(self, example: Dict, padding="max_length") -> Dict:
        """
        Args:
            example (Dict):
                A dictionary of one sample in the PDFData with the following fields:
                    "words": List[str]
                    "labels": List[str]
                    "bbox": List[List]
                    "block_ids": List[int]
                    "line_ids": List[int]
        """
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
                    encoded_word_ids.append(word_idx)

            batched_labels.append(cur_label_ids)
            batched_bboxes.append(cur_bboxes)

            # Find the last word id in this batch to handle
            # multi-batch samples
            for word_id in reversed(word_ids):
                if word_id is not None:
                    previous_word_idx = word_id
                    break

        tokenized_inputs["labels"] = batched_labels
        tokenized_inputs["bbox"] = batched_bboxes
        tokenized_inputs["encoded_word_ids"] = list(set(encoded_word_ids))
        return tokenized_inputs

    def preprocess_batch(self, examples: Dict[str, List]) -> Dict[str, List]:
        """This is a wrapper based on the preprocess_sample. There might be
        considerable performance loss, but we don't need to rewrite the main
        iteration loop again.
        """
        all_processed_examples = []
        for example in self.iter_example(examples):
            processed_example = self.preprocess_sample(example, padding=False)
            processed_example.pop("encoded_word_ids")
            # encoded_word_ids will only be used in test time
            all_processed_examples.append(processed_example)

        return self.batchsize_examples(all_processed_examples)