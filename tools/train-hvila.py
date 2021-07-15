# Modified based on https://github.com/huggingface/transformers/tree/master/examples/token-classification

import logging
import os
import sys
import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from utils import (
    load_json,
    write_json,
    classification_report,
    DataCollatorForTokenClassification,
    columns_used_in_model_inputs,
)

from arguments import *

sys.path.append("..")
from vila.models.hierarchical_model import HierarchicalModelForTokenClassification
from vila.models.configuration_hierarchical_model import HierarchicalModelConfig
from vila.dataset.preprocessors import instantiate_dataset_preprocessor
from vila.constants import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)


def flatten_line_level_prediction(batched_line_pred, batched_line_word_count):
    final_flattend_pred = []
    for line_pred, line_word_count in zip(batched_line_pred, batched_line_word_count):
        assert len(line_pred) == len(line_word_count)
        for (pred, label), (line_id, count) in zip(line_pred, line_word_count):
            final_flattend_pred.append([[pred, label, line_id]] * count)

    return list(itertools.chain.from_iterable(final_flattend_pred))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        filename="./log.txt",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.validation_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "words"
    label_column_name = "labels"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if data_args.label_map_file is not None:
        label_list = list(load_json(data_args.label_map_file).values())
        label_to_id = {l: i for i, l in enumerate(label_list)}
    else:
        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(datasets["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
            label_list = [str(ele) for ele in label_list]  # Ensure the ele is a string

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = HierarchicalModelConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        textline_encoder_output=model_args.textline_encoder_output,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = HierarchicalModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    logger.info(f"The used agg level is {data_args.agg_level}")
    preprocessor = instantiate_dataset_preprocessor(
        "hierarchical_modeling", tokenizer, data_args
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocessor.preprocess_batch,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocessor.preprocess_sample,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    def data_collator(batch):
        condensed_batch = {
            key: torch.tensor([ele[key] for ele in batch], dtype=torch.int64)
            for key in batch[0].keys()
        }
        assert not (
            condensed_batch["group_level_attention_mask"][
                condensed_batch["labels"] == -100
            ]
            > 0
        ).any()
        return condensed_batch

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_predictions = list(itertools.chain.from_iterable(true_predictions))
        true_labels = list(itertools.chain.from_iterable(true_labels))
        results = classification_report(y_true=true_labels, y_pred=true_predictions)
        results.pop("detailed")
        return results

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        # Disable this behavior as we are re-initializing the classifier weights
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = (
            data_args.max_val_samples
            if data_args.max_val_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        output_test_predictions_file = os.path.join(
            training_args.output_dir, data_args.pred_file_name
        )
        logger.info(f"The test file will be saved to {output_test_predictions_file}")

        _used_cols = columns_used_in_model_inputs(test_dataset, model)
        _used_cols = [ele for ele in _used_cols if "label" not in ele]

        all_pred_df = []

        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_dataset)):
                _sample = {
                    key: torch.tensor(val, dtype=torch.int64, device=model.device)
                    for key, val in sample.items()
                    if key in _used_cols
                }
                model_outputs = model(**_sample)
                predictions = model_outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = sample["labels"]

                true_predictions = [
                    [(p, l) for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                flatten_predictions = flatten_line_level_prediction(
                    true_predictions, sample["group_word_count"]
                )
                assert len(flatten_predictions) == len(sample["words"])
                df = pd.DataFrame(
                    flatten_predictions, columns=["pred", "gt", "line_id"]
                )
                df["word"] = sample["words"]
                df["block_id"] = sample["block_ids"]
                df["word_id"] = df.index
                df["sample_id"] = idx
                all_pred_df.append(df)

            pd.concat(all_pred_df).to_csv(output_test_predictions_file)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
