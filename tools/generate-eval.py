from typing import List, Union, Dict, Any, Tuple
import os
import json
from glob import glob
from dataclasses import dataclass
import functools
import argparse

from sklearn import metrics
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from datasets import ClassLabel, load_dataset, load_metric

from utils import *
from dataset_configs import *

tqdm.pandas()  # enable progress_apply

def flatten_dataset_to_table(dataset) -> pd.DataFrame:
    """Convert the HF Dataset to a Pandas DataFrame"""

    results = []

    for e_id, example in enumerate(tqdm(dataset)):
        cur_len = len(example["words"])
        results.extend(
            [
                [
                    e_id,
                    i,
                    example["words"][i],
                    example["labels"][i],
                    example["block_ids"][i],
                    example["line_ids"][i],
                ]
                for i in range(cur_len)
            ]
        )

    return pd.DataFrame(
        results,
        columns=["sample_id", "word_id", "word", "label", "block_id", "line_id"],
    )


def load_dataset_and_flatten(dataset_path) -> pd.DataFrame:
    if os.path.exists(dataset_path.replace(".json", ".cached.csv")):
        return pd.read_csv(dataset_path.replace(".json", ".cached.csv"))
    else:
        dataset = load_dataset("json", data_files=dataset_path, field="data")
        df = flatten_dataset_to_table(dataset["train"])
        df.to_csv(dataset_path.replace(".json", ".cached.csv"), index=None)
        return df


def _preprocess_prediction_table(
    test_df, pred_df, most_frequent_category=None, label_mapping: Dict = None
) -> pd.DataFrame:
    """Merge the prediction table with the original gt table
    to 1) fetch the gt and 2) insert some "un-tokenized" tokens
    """

    merged_df = test_df.merge(
        pred_df.loc[:, ["sample_id", "word_id", "pred"]],
        how="outer",
        on=["sample_id", "word_id"],
    )

    if label_mapping is not None:
        merged_df["pred"] = merged_df["pred"].map(label_mapping)

    if most_frequent_category is None:
        most_frequent_category = test_df["label"].value_counts().index[0]

    merged_df["pred"] = merged_df["pred"].fillna(
        most_frequent_category
    )  # fill in the most frequent category

    return merged_df


@dataclass
class ModelConfig:
    task_name: str = ""
    model_name: str = ""
    variant: str = ""


def put_model_config_at_the_first(func):
    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
        df = func(self, *args, **kwargs)
        columns = df.columns
        return df[["task_name", "model_name", "variant"] + list(columns[:-3])]

    return wrap


class SingleModelPrediction:
    """Methods for processing the "test_predictions" tables for an individual model"""

    def __init__(
        self,
        df,
        label_space,
        model_config: ModelConfig,
        gt_name="label",
        pred_name="pred",
        used_metric="entropy",
    ):

        self.df = df
        self.label_space = label_space
        self.gt_name = gt_name
        self.pred_name = pred_name

        self.model_config = model_config
        self.used_metric = used_metric

    @classmethod
    def from_raw_prediction_table(
        cls,
        test_df,
        pred_df,
        label_space,
        model_config,
        most_frequent_category: int = None,
        label_mapping=None,
        used_metric="entropy",
        **kwargs,
    ):
        merged_df = _preprocess_prediction_table(
            test_df,
            pred_df,
            most_frequent_category,
            label_mapping=label_mapping,
        )
        return cls(
            merged_df, label_space, model_config, used_metric=used_metric, **kwargs
        )

    def groupby(self, level):

        assert level in ["block", "line"]
        return self.df.groupby(["sample_id", f"{level}_id"])

    def calculate_per_category_scores(self):
        _scores = precision_recall_fscore_support(
            self.df[self.gt_name],
            self.df[self.pred_name],
            labels=self.label_space,
            zero_division=0,
        )
        _scores = pd.DataFrame(
            _scores,
            columns=self.label_space,
            index=["precision", "recall", "f-score", "support"],
        )
        return _scores

    def calculate_accuracy_for_group(self, gp, score_average="micro"):

        accuracy = (gp[self.gt_name] == gp[self.pred_name]).mean()

        precision, recall, fscore, _ = precision_recall_fscore_support(
            gp[self.gt_name],
            gp[self.pred_name],
            average=score_average,
            labels=self.label_space,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }

    def calculate_gini_score_for_group(self, gp):

        cts = gp[self.pred_name].value_counts()
        if len(cts) == 1:
            return 0
        else:
            return 1 - ((cts / cts.sum()) ** 2).sum()

    def calculate_entropy_for_group(self, gp):

        cts = gp[self.pred_name].value_counts()
        if len(cts) == 1:
            return 0
        else:
            prob = cts / cts.sum()
            entropy = -(prob * np.log2(prob)).sum()
            return entropy

    def create_page_level_accuracy_report(self) -> pd.DataFrame:

        return (
            self.df.groupby("sample_id")
            .apply(self.calculate_accuracy_for_group)
            .apply(pd.Series)
        )

    def create_page_level_gini_report(self, level="block") -> pd.Series:

        gini = self.groupby(level=level).apply(self.calculate_gini_score_for_group)
        gini = (
            gini.to_frame()
            .rename(columns={0: "gini"})
            .reset_index()
            .groupby("sample_id")
            .gini.mean()
        )
        return gini

    def create_page_level_entropy_report(self, level="block") -> pd.Series:

        entropy = self.groupby(level=level).apply(self.calculate_entropy_for_group)
        entropy = (
            entropy.to_frame()
            .rename(columns={0: "entropy"})
            .reset_index()
            .groupby("sample_id")
            .entropy.mean()
        )
        return entropy

    def create_page_level_ami_report(self) -> pd.DataFrame:

        ami = (
            self.df.groupby("sample_id")
            .apply(
                lambda gp: metrics.adjusted_mutual_info_score(
                    gp[self.gt_name], gp[self.pred_name]
                )
            )
            .to_frame()
            .rename(columns={0: "ami"})
        )
        return ami

    def create_page_level_overall_report(self) -> pd.DataFrame:

        report = self.create_page_level_accuracy_report()
        report["gini"] = self.create_page_level_gini_report()
        report["entropy"] = self.create_page_level_entropy_report()
        return report

    def create_all_page_accuracy_report(self) -> pd.Series:
        return pd.Series(
            self.calculate_accuracy_for_group(self.df, score_average="macro")
        )

    def create_all_page_ami_report(self) -> pd.Series:
        return pd.Series(self.create_page_level_ami_report().mean())

    def create_all_page_gini_report(self, level="block") -> pd.Series:

        gini = self.create_page_level_gini_report(level=level)
        report = pd.Series(
            {
                f"gini_{level}_average": gini.mean(),
                f"gini_{level}_std": gini.std(),
                f"gini_{level}_nonzero": gini[gini > 0].count(),
            }
        )
        return report

    def create_all_page_entropy_report(self, level="block") -> pd.Series:

        entropy = self.create_page_level_entropy_report(level=level)
        report = pd.Series(
            {
                f"entropy_{level}_average": entropy.mean(),
                f"entropy_{level}_std": entropy.std(),
                f"entropy_{level}_nonzero": entropy[entropy > 0].count(),
            }
        )
        return report

    def create_all_page_overall_report(self, add_line_level_gini=False) -> pd.Series:
        report = self.create_all_page_accuracy_report()

        if self.used_metric == "gini":
            gini = self.create_all_page_gini_report()
            if add_line_level_gini:
                gini = gini.append(self.create_all_page_gini_report(level="line"))
            report = report.append(gini)

        elif self.used_metric == "entropy":
            entropy = self.create_all_page_entropy_report()
            if add_line_level_gini:
                entropy = entropy.append(
                    self.create_all_page_entropy_report(level="line")
                )
            report = report.append(entropy)

        report = report.append(self.create_all_page_ami_report())

        return report

    def majority_voting_postprocessing(self, level) -> "SingleModelPrediction":
        """This method attempts to use majority voting for model predictions within each
        group (level) to improve the accuracy. It will firstly use groupby the elements
        within each group, then find the most common class in the predicted categoires,
        and replace the others as the predicted category.
        """
        # It might take a while
        df = (
            self.groupby(level=level)
            .progress_apply(
                lambda gp: gp.assign(pred=gp[self.pred_name].value_counts().index[0])
            )
            .reset_index(drop=True)
        )

        return self.__class__(
            df,
            **{key: getattr(self, key) for key in self.__dict__.keys() if key != "df"},
        )


@dataclass
class MultiModelPrediction:
    """Methods for processing the "test_predictions" tables for multiple models
    within a Experiment
    """

    predictions: List[SingleModelPrediction]
    name: str

    def create_per_category_report(self) -> pd.DataFrame:

        reports = []
        for prediction in self.predictions:
            report = prediction.calculate_per_category_scores()
            report["task_name"] = prediction.model_config.task_name
            report["model_name"] = prediction.model_config.model_name
            report["variant"] = prediction.model_config.variant
            reports.append(report)
        return pd.concat(reports)

    @put_model_config_at_the_first
    def create_overall_report(self) -> pd.DataFrame:
        reports = []
        for prediction in self.predictions:
            report = prediction.create_all_page_overall_report()
            report["task_name"] = prediction.model_config.task_name
            report["model_name"] = prediction.model_config.model_name
            report["variant"] = prediction.model_config.variant
            reports.append(report)
        return pd.DataFrame(reports)

    @put_model_config_at_the_first
    def create_overall_report_with_majority_voting_postprocessing(
        self, level
    ) -> pd.DataFrame:
        reports = []
        for prediction in self.predictions:
            report = prediction.majority_voting_postprocessing(
                level=level
            ).create_all_page_overall_report()
            report["task_name"] = prediction.model_config.task_name
            report["model_name"] = prediction.model_config.model_name
            report["variant"] = prediction.model_config.variant
            reports.append(report)
        return pd.DataFrame(reports)

    @classmethod
    def from_experiment_folder(
        cls,
        experiment_folder,
        test_df,
        label_space,
        experiment_name=None,
        most_frequent_category=None,
        label_mapping=None,
        used_metric="entropy",
        prediction_filename="test_predictions.csv",
    ):

        if experiment_name is None:
            experiment_name = os.path.basename(experiment_folder)

        predictions = []
        model_names = glob(f"{experiment_folder}/*")

        for model_name in tqdm(model_names):
            try:
                df = pd.read_csv(f"{model_name}/{prediction_filename}")
                model_name = os.path.basename(model_name)
                model_config = ModelConfig(
                    task_name=experiment_name,
                    model_name=model_name,
                )
                predictions.append(
                    SingleModelPrediction.from_raw_prediction_table(
                        test_df=test_df,
                        pred_df=df,
                        label_space=label_space,
                        model_config=model_config,
                        most_frequent_category=most_frequent_category,
                        label_mapping=label_mapping,
                        used_metric=used_metric,
                    )
                )
            except:
                print(f"Error loading for {model_name} in MultiModelPrediction")

        return cls(predictions, experiment_name)


class SingleModelRecord:
    """Methods for processing training records for a single model"""

    def __init__(
        self,
        model_folder,
        model_config,
        trainer_states_name="trainer_state.json",
        all_results_name="all_results.json",
        training_args_name="training_args.bin",
    ):
        self.model_config = model_config

        self.trainer_states = load_json(f"{model_folder}/{trainer_states_name}")
        self.all_results = load_json(f"{model_folder}/{all_results_name}")
        self.training_args = torch.load(
            f"{model_folder}/{training_args_name}"
        ).to_dict()

    def load_acc_history(self) -> pd.DataFrame:
        cur_report = []
        for ele in self.trainer_states["log_history"]:
            cur_report.append(
                [
                    ele["step"],
                    ele["epoch"],
                    ele.get("eval_fscore"),
                    ele.get("eval_accuracy"),
                ]
            )
        df = pd.DataFrame(cur_report, columns=["step", "epoch", "f1-score", "acc"])
        return df

    def load_loss_history(self) -> pd.DataFrame:
        cur_report = []
        for ele in self.trainer_states["log_history"]:
            if "loss" in ele:
                cur_report.append([ele["step"], ele["epoch"], ele["loss"]])
        df = pd.DataFrame(cur_report, columns=["step", "epoch", "loss"])
        return df

    def load_train_history(self) -> pd.DataFrame:
        acc_record = self.load_acc_history()
        loss_record = self.load_loss_history()
        merged = acc_record.merge(loss_record, how="outer")
        return merged

    def load_computation_record(self) -> pd.Series:

        return pd.Series(
            {
                "gpus": self.training_args["_n_gpu"],
                "batch_size": self.training_args["per_device_train_batch_size"],
                "epochs": self.training_args["num_train_epochs"],
                "learning_rate": self.training_args["learning_rate"],
                "warmup_steps": self.training_args["warmup_steps"],
                "train_samples": self.all_results["train_samples"],
                "train_flos": self.trainer_states["total_flos"],
                "train_steps": self.trainer_states["max_steps"],
                "train_runtime": self.all_results["train_runtime"],
                "eval_runtime": self.all_results["eval_runtime"],
                "eval_samples": self.all_results["eval_samples"],
                "eval_samples_per_second": self.all_results["eval_samples_per_second"],
                "eval_fscore": self.all_results["eval_fscore"],
            }
        )


@dataclass
class MultiModelRecord:
    records: List[SingleModelRecord]
    name: str

    @put_model_config_at_the_first
    def load_train_history(self) -> pd.DataFrame:
        reports = []
        for record in self.records:
            report = record.load_train_history()
            report["task_name"] = record.model_config.task_name
            report["model_name"] = record.model_config.model_name
            report["variant"] = record.model_config.variant
            reports.append(report)
        return pd.concat(reports)

    @put_model_config_at_the_first
    def load_computation_record(self) -> pd.DataFrame:
        reports = []
        for record in self.records:
            report = record.load_computation_record()
            report["task_name"] = record.model_config.task_name
            report["model_name"] = record.model_config.model_name
            report["variant"] = record.model_config.variant
            reports.append(report)
        return pd.DataFrame(reports)

    @classmethod
    def from_experiment_folder(
        cls,
        experiment_folder,
        experiment_name=None,
        trainer_states_name="trainer_state.json",
        all_results_name="all_results.json",
        training_args_name="training_args.bin",
    ):

        if experiment_name is None:
            experiment_name = os.path.basename(experiment_folder)

        records = []
        model_names = glob(f"{experiment_folder}/*")

        for model_name in tqdm(model_names):

            try:
                model_config = ModelConfig(
                    task_name=experiment_name, model_name=os.path.basename(model_name)
                )
                records.append(
                    SingleModelRecord(
                        model_name,
                        model_config,
                        trainer_states_name=trainer_states_name,
                        all_results_name=all_results_name,
                        training_args_name=training_args_name,
                    )
                )
            except:
                print(f"Error loading for {model_name}")

        return cls(records, experiment_name)


@dataclass
class CombinedReport:
    records: MultiModelRecord
    predictions: MultiModelPrediction

    def report(self, with_majority_voting=True) -> pd.DataFrame():

        computational_report = self.records.load_computation_record()
        scores_report = self.predictions.create_overall_report()

        if not with_majority_voting:

            return computational_report.merge(
                scores_report, on=["task_name", "model_name", "variant"]
            ).set_index(["task_name", "model_name", "variant"])

        else:
            scores_report_with_majority_voting = self.predictions.create_overall_report_with_majority_voting_postprocessing(
                level="block"
            )

            return (
                computational_report.merge(
                    scores_report, on=["task_name", "model_name", "variant"]
                )
                .merge(
                    scores_report_with_majority_voting,
                    on=["task_name", "model_name", "variant"],
                    suffixes=("", "_majority_voting"),
                )
                .set_index(["task_name", "model_name", "variant"])
            )

    def report_per_category_scores(
        self, column_names: Union[List, Dict] = None
    ) -> pd.DataFrame:

        scores_report = self.predictions.create_per_category_report()
        if column_names is not None:
            if isinstance(column_names, list):
                scores_report.columns = column_names
            elif isinstance(column_names, dict):
                scores_report.columns = [
                    column_names.get(col, col) for col in scores_report.columns
                ]

        return scores_report.reset_index().set_index(
            ["task_name", "model_name", "variant", "index"]
        )

    @classmethod
    def from_experiment_folder(
        cls,
        experiment_folder,
        test_df,
        label_space,
        experiment_name=None,
        most_frequent_category=None,
        prediction_filename="test_predictions.csv",
        trainer_states_name="trainer_state.json",
        all_results_name="all_results.json",
        training_args_name="training_args.bin",
        used_metric="entropy",
    ):

        predictions = MultiModelPrediction.from_experiment_folder(
            experiment_folder,
            test_df,
            label_space,
            experiment_name=experiment_name,
            most_frequent_category=most_frequent_category,
            prediction_filename=prediction_filename,
            used_metric=used_metric,
        )

        records = MultiModelRecord.from_experiment_folder(
            experiment_folder,
            experiment_name=experiment_name,
            trainer_states_name=trainer_states_name,
            all_results_name=all_results_name,
            training_args_name=training_args_name,
        )

        return cls(records, predictions)

def generate_eval_report_for_experiment(experiment_folder, args):
    
    assert os.path.isdir(experiment_folder), f"{experiment_folder} does not exist"
    print(f"Working on generating experiment results for {experiment_folder}")

    dataset = instiantiate_dataset(args.dataset_name)
    test_df = load_dataset_and_flatten(dataset.test_file)

    all_labels = load_json(dataset.label_map_file)
    label2id = {val:int(key) for key, val in all_labels.items()}
    id2label = {int(key):val for key, val in all_labels.items()}
    dataset_labels = list(id2label.keys())

    print(f"Loading from {experiment_folder}")
    report = CombinedReport.from_experiment_folder(
        experiment_folder,
        test_df=test_df,
        label_space=dataset_labels,
    )

    report_folder = os.path.join(experiment_folder, args.report_folder_name)
    os.makedirs(report_folder, exist_ok=True)
    report_df = report.report()
    report_df.to_csv(os.path.join(report_folder, "report.csv"))

    if args.store_per_class:
        report_df_per_cat = report.report_per_category_scores()
        report_df_per_cat.to_csv(os.path.join(report_folder, "report_per_class.csv"))
        return report_df, report_df_per_cat
    
    return report_df, None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='The name of the used dataset')
    parser.add_argument('--base_path', default="../checkpoints", help='The checkpoint base path')
    parser.add_argument('--experiment_name', default=None, type=str, help='The name of the experiment.')
    parser.add_argument('--store_per_class', action='store_true', help='Store per class accuracy scores.')

    parser.add_argument('--report_folder_name', default="_reports", help='The name of the folder for saving reports')

    args = parser.parse_args()
    
    dataset_path = os.path.join(args.base_path, args.dataset_name.lower())

    if args.experiment_name is not None:
        experiment_folder = os.path.join(dataset_path, args.experiment_name)
        generate_eval_report_for_experiment(experiment_folder, args)
    else:
        print(f"No experiment_name is specified, iterating all the experiment folders in {args.base_path=}")
        
        all_report_df = []
        all_report_df_per_cat = []

        for experiment_name in os.listdir(dataset_path):
            
            if not experiment_name.startswith(".") and experiment_name != args.report_folder_name:
                experiment_folder = os.path.join(dataset_path, experiment_name)
                report_df, report_df_per_cat = generate_eval_report_for_experiment(experiment_folder, args)
                
                all_report_df.append(report_df)
                all_report_df_per_cat.append(report_df_per_cat)
        
        all_report_df = pd.concat(all_report_df)
        
        report_folder = os.path.join(dataset_path, args.report_folder_name)
        os.makedirs(report_folder, exist_ok=True)
        all_report_df.to_csv(os.path.join(report_folder, "report.csv"))

        if args.store_per_class:
            all_report_df_per_cat = pd.concat(all_report_df_per_cat)
            all_report_df_per_cat.to_csv(os.path.join(report_folder, "report_per_class.csv"))