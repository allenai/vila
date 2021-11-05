from typing import List, Union, Dict, Any, Tuple
import random
import argparse
import json
import itertools
import os
from glob import glob
from dataclasses import dataclass
from collections import defaultdict

from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import layoutparser as lp

PADDING_CONSTANT = 10000

np.random.seed(42)
random.seed(42)


def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def write_json(data, filename):
    with open(filename, "w") as fp:
        json.dump(data, fp)


def cvt_df_to_layout(row):

    return lp.TextBlock(
        lp.Rectangle(
            row["x_1"],
            row["y_1"],
            row["x_2"],
            row["y_2"],
        ),
        id=row["id"],
        type=row["category"],
        text=row["text"],
    )


class RawAnnotation:
    def __init__(self, annotation_table, annotation_dir):

        self.annotation_table = pd.read_csv(annotation_table).set_index("sha")
        self.annotation_dir = annotation_dir

    def load_annotation_for_sha(self, sha):

        all_page_annotations = {}
        
        if len(glob(f"{self.annotation_dir}/{sha}-*.json"))>0:
            # Load annotation for sha-pageid.json like files 
            for filename in glob(f"{self.annotation_dir}/{sha}-*.json"):
                page_id = int(filename.replace(f"{self.annotation_dir}/{sha}-", "").replace(".json",""))
                res = self.load_page_data_from_json(filename)
                if res is not None:
                    all_page_annotations[page_id] = res
        else:
            # load annotations for sha.json like files 
            for filename in glob(f"{self.annotation_dir}/{sha}.json"):
                all_page_annotations = self.load_all_page_data_from_json(filename)

        return all_page_annotations

    def load_all_page_data_from_json(self, filename):
        raw = load_json(filename)
        results_by_page = defaultdict(list)
        for ele in raw["annotations"]:

            results_by_page[ele["page"]].append(
                lp.TextBlock(
                    lp.Rectangle(
                        ele["bounds"]["left"],
                        ele["bounds"]["top"],
                        ele["bounds"]["right"],
                        ele["bounds"]["bottom"],
                    ),
                    type=ele["label"]["text"],
                )
            )
        return results_by_page

    def load_page_data_from_json(self, filename):
        raw = load_json(filename)
        page_annotation = []
        for ele in raw["annotations"]:
            page_annotation.append(
                lp.TextBlock(
                    lp.Rectangle(
                        ele["bounds"]["left"],
                        ele["bounds"]["top"],
                        ele["bounds"]["right"],
                        ele["bounds"]["bottom"],
                    ),
                    type=ele["label"]["text"],
                )
            )
        return page_annotation


@dataclass
class PageData:
    blocks: List[lp.TextBlock]
    lines: List[lp.TextBlock]
    words: List[lp.TextBlock]


class CERMINEAnnotation:
    def __init__(
        self,
        pdf_directory,
        csv_directory,
    ):
        self.csv_dir = csv_directory
        self.pdf_dir = pdf_directory

    @staticmethod
    def load_page_data_from_csv(filename):
        df = pd.read_csv(filename)
        if len(df) == 0:
            return None

        df = df[~df.text.isna()]
        if len(df) == 0:
            return None

        blocks_df = df[df.is_block]
        lines_df = df[df.is_line]
        tokens_df = df[~df.is_line & ~df.is_block]

        return PageData(
            blocks=lp.Layout(blocks_df.apply(cvt_df_to_layout, axis=1).tolist()),
            lines=lp.Layout(lines_df.apply(cvt_df_to_layout, axis=1).tolist()),
            words=lp.Layout(tokens_df.apply(cvt_df_to_layout, axis=1).tolist()),
        )

    def load_annotations_for_sha(self, sha):

        xml_data = {}
        for filename in glob(f"{self.csv_dir}/{sha}-*.csv"):
            page_id = int(filename.replace(f"{self.csv_dir}/{sha}-", "").replace(".csv",""))
            res = self.load_page_data_from_csv(filename)
            if res is not None:
                xml_data[page_id] = res

        return xml_data


class VISIONAnnotation(CERMINEAnnotation):
    @staticmethod
    def load_page_data_from_csv(filename):
        df = pd.read_csv(filename)
        if len(df) == 0:
            return None
        # Not dropping empty tokens 

        blocks_df = df[df.is_block]
        lines_df = df[df.is_line]
        tokens_df = df[~df.is_line & ~df.is_block]

        return PageData(
            blocks=lp.Layout(blocks_df.apply(cvt_df_to_layout, axis=1).tolist()),
            lines=lp.Layout(lines_df.apply(cvt_df_to_layout, axis=1).tolist()),
            words=lp.Layout(tokens_df.apply(cvt_df_to_layout, axis=1).tolist()),
        )

class S2VLAnnotationGenerator:
    def __init__(
        self,
        annotation_table,
        raw_annotation,
        cermine_annotation,
        selected_categories,
        default_category,
        vision_annotation=None,
    ):
        self.annotation_table = pd.read_csv(annotation_table)
        self.raw_annotation = raw_annotation
        self.cermine_annotation = cermine_annotation
        self.vision_annotation = vision_annotation

        self.selected_categories = selected_categories
        self.default_category = default_category
        self.cat2id = {cat: idx for idx, cat in enumerate(selected_categories)}
        self.id2cat = {idx: cat for idx, cat in enumerate(selected_categories)}

    def get_unique_shas(self):
        return self.annotation_table.sha.unique()

    def convert_token_data_to_json(self, tokens):
        token_df = pd.DataFrame(
            [
                [
                    e.id,
                    str(e.text),
                    [int(_) for _ in e.coordinates],
                    e.type,
                    e.block_id,
                    e.line_id,
                ]
                for e in tokens
            ],
            columns=["id", "text", "bbox", "category", "block_id", "line_id"],
        )
        token_df = token_df[
            ~token_df.text.isnull()
            & ~token_df.text.isna()
            & ~token_df.text.str.isspace()
        ]
        row_item = {
            "words": token_df["text"].tolist(),
            "bbox": token_df["bbox"].tolist(),
            "labels": token_df["category"].map(self.cat2id).tolist(),
            "block_ids": token_df["block_id"].astype("int").tolist(),
            "line_ids": token_df["line_id"].astype("int").tolist(),
        }

        return row_item

    def create_annotation_for_sha(self, sha):

        all_token_data = []
        all_files = []

        raw_blocks = self.raw_annotation.load_annotation_for_sha(sha)
        cermine_data = self.cermine_annotation.load_annotations_for_sha(sha)

        for page_id in cermine_data.keys():
            blocks = [
                b for b in raw_blocks[page_id] if b.type in self.selected_categories
            ]

            # Pass 1: O(n) Initialize ids and categories
            for word in cermine_data[page_id].words:
                word.line_id = -1
                word.block_id = -1
                word.type = self.default_category

            # Pass 2: O(mn) Assign token categories
            for word in cermine_data[page_id].words:
                for block in blocks:
                    if word.is_in(block, center=True):
                        word.type = block.type

            # Pass 3: O(mn) Assign token block-category ids
            used_lines_for_assign_line_ids = cermine_data[page_id].lines
            used_blocks_for_assign_block_ids = cermine_data[page_id].blocks
            for word in cermine_data[page_id].words:
                for _l in used_lines_for_assign_line_ids:
                    if word.is_in(_l, center=True):
                        word.line_id = _l.id

                for _b in used_blocks_for_assign_block_ids:
                    if word.is_in(_b, center=True):
                        word.block_id = _b.id

            # Pass 4: O(n) In case some blocks are not assigned with the
            # appropriate block indices, we assign the line ids
            for word in cermine_data[page_id].words:
                if word.block_id == -1:
                    word.block_id = word.line_id + PADDING_CONSTANT

            row_item = self.convert_token_data_to_json(cermine_data[page_id].words)

            if len(row_item["words"]) > 0:

                all_token_data.append(row_item)
                all_files.append(f"{sha}-{page_id}")

        return all_token_data, all_files

    def create_annotation_for_shas(self, shas):
        all_token_data = []
        all_files = []
        pbar = tqdm(shas)
        for sha in pbar:
            pbar.set_description(sha)
            token_data, files = self.create_annotation_for_sha(sha)
            all_token_data.extend(token_data)
            all_files.extend(files)
        return all_token_data, all_files

    def create_annotations(self):
        shas = self.get_unique_shas()
        all_token_data, all_files = self.create_annotation_for_shas(shas)
        all_valid_shas = list(set([ele.split("-")[0] for ele in all_files]))

        self.all_token_data = all_token_data
        self.all_files = all_files
        self.all_valid_shas = all_valid_shas
        self.sha_to_sample_mapping = {
            sha: [idx for idx, file in enumerate(all_files) if file[:40] == sha]
            for sha in all_valid_shas
        }

    def save_annotation_cv(self, export_folder, n_fold=5):

        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

        for idx, (train_idx, test_idx) in enumerate(
            tqdm(kf.split(self.all_valid_shas), total=n_fold)
        ):
            annotation_data = {}
            train_test_split = {}

            for name, indices in [("train", train_idx), ("test", test_idx)]:
                cur_shas = [self.all_valid_shas[i] for i in indices]
                selected_data_item_indices = list(
                    itertools.chain.from_iterable(
                        [self.sha_to_sample_mapping[sha] for sha in cur_shas]
                    )
                )

                annotation_data[name] = (
                    [self.all_token_data[i] for i in selected_data_item_indices],
                    [self.all_files[i] for i in selected_data_item_indices],
                )
                train_test_split[name] = annotation_data[name][1]

            cur_export_folder = f"{export_folder}/{idx}"
            self.save_json(annotation_data, train_test_split, cur_export_folder)

    def save_annotation_few_shot(self, export_folder, sample_sizes=[5, 10, 15]):

        for sample_size in tqdm(sample_sizes):

            train_sha, test_sha = train_test_split(
                self.all_valid_shas, train_size=sample_size, random_state=42
            )

            annotation_data = {}
            train_test_files = {}

            for name, cur_shas in [("train", train_sha), ("test", test_sha)]:
                selected_data_item_indices = list(
                    itertools.chain.from_iterable(
                        [self.sha_to_sample_mapping[sha] for sha in cur_shas]
                    )
                )

                annotation_data[name] = (
                    [self.all_token_data[i] for i in selected_data_item_indices],
                    [self.all_files[i] for i in selected_data_item_indices],
                )
                train_test_files[name] = annotation_data[name][1]

            cur_export_folder = f"{export_folder}/{sample_size}"
            self.save_json(annotation_data, train_test_files, cur_export_folder)

    def save_annotation_few_shot_with_mutual_test_set(
        self, export_folder, sample_sizes=[5, 10, 15]
    ):

        maximum_training_samples = max(sample_sizes)
        maximum_remaining_test_samples = (
            len(self.all_valid_shas) - maximum_training_samples
        )

        all_train_sha, test_sha = train_test_split(
            self.all_valid_shas,
            test_size=maximum_remaining_test_samples,
            random_state=42,
        )

        for sample_size in tqdm(sample_sizes):

            train_sha = random.sample(all_train_sha, sample_size)

            annotation_data = {}
            train_test_files = {}

            for name, cur_shas in [("train", train_sha), ("test", test_sha)]:
                selected_data_item_indices = list(
                    itertools.chain.from_iterable(
                        [self.sha_to_sample_mapping[sha] for sha in cur_shas]
                    )
                )

                annotation_data[name] = (
                    [self.all_token_data[i] for i in selected_data_item_indices],
                    [self.all_files[i] for i in selected_data_item_indices],
                )
                train_test_files[name] = annotation_data[name][1]

            cur_export_folder = f"{export_folder}/{sample_size}"
            self.save_json(annotation_data, train_test_files, cur_export_folder)

    def save_annotation_few_shot_and_cv(
        self, export_folder, train_test_shas, sample_sizes=[5, 10, 15, 25, 45, 70]
    ):

        for cv_index, _shas in enumerate(tqdm(train_test_shas)):
            all_train_sha, test_sha = _shas["train"], _shas["test"]
            for sample_size in sample_sizes:
                train_sha = all_train_sha[:sample_size]

                annotation_data = {}
                train_test_files = {}

                for name, cur_shas in [("train", train_sha), ("test", test_sha)]:
                    selected_data_item_indices = list(
                        itertools.chain.from_iterable(
                            [self.sha_to_sample_mapping[sha] for sha in cur_shas]
                        )
                    )

                    annotation_data[name] = (
                        [self.all_token_data[i] for i in selected_data_item_indices],
                        [self.all_files[i] for i in selected_data_item_indices],
                    )
                    train_test_files[name] = annotation_data[name][1]

                cur_export_folder = f"{export_folder}/{sample_size}/{cv_index}"
                self.save_json(annotation_data, train_test_files, cur_export_folder)

    def save_json(self, annotation_data, train_test_split, export_folder):

        os.makedirs(export_folder, exist_ok=True)

        for subset, (all_token_data, all_files) in annotation_data.items():

            write_json(
                {"data": all_token_data, "labels": self.cat2id, "files": all_files},
                f"{export_folder}/{subset}-token.json",
            )

        write_json(train_test_split, f"{export_folder}/train-test-split.json")
        write_json(self.id2cat, f"{export_folder}/labels.json")


class S2VLAnnotationGeneratorWithGTBox(S2VLAnnotationGenerator):
    @staticmethod
    def order_blocks_based_on_token_ids(blocks, tokens):

        token_ids_in_blocks = []

        for block in blocks:

            token_ids_in_this_block = []

            for token in tokens:
                if token.is_in(block, center=True):
                    token_ids_in_this_block.append(token.id)

            if len(token_ids_in_this_block) == 0:
                token_ids_in_blocks.append(float("inf"))
            else:
                token_ids_in_blocks.append(min(token_ids_in_this_block))

        sorted_blocks = [
            x.set(id=idx)
            for idx, (_, x) in enumerate(
                sorted(zip(token_ids_in_blocks, blocks), key=lambda pair: pair[0])
            )
        ]

        return sorted_blocks

    def create_annotation_for_sha(self, sha):

        all_token_data = []
        all_files = []

        raw_blocks = self.raw_annotation.load_annotation_for_sha(sha)
        cermine_data = self.cermine_annotation.load_annotations_for_sha(sha)

        for page_id in cermine_data.keys():
            blocks = [
                b for b in raw_blocks[page_id] if b.type in self.selected_categories
            ]
            blocks = self.order_blocks_based_on_token_ids(
                blocks, cermine_data[page_id].words
            )

            # Pass 1: O(n) Initialize ids and categories
            for word in cermine_data[page_id].words:
                word.line_id = -1
                word.block_id = -1
                word.type = self.default_category

            # Pass 2: O(mn) Assign token categories
            for word in cermine_data[page_id].words:
                for block in blocks:
                    if word.is_in(block, center=True):
                        word.type = block.type

            # Pass 3: O(mn) Assign token block-category ids
            used_lines_for_assign_line_ids = cermine_data[page_id].lines
            used_blocks_for_assign_block_ids = blocks

            for word in cermine_data[page_id].words:
                for _l in used_lines_for_assign_line_ids:
                    if word.is_in(_l, center=True):
                        word.line_id = _l.id

                for _b in used_blocks_for_assign_block_ids:
                    if word.is_in(_b, center=True):
                        word.block_id = _b.id

            # Pass 4: O(n) In case some blocks are not assigned with the
            # appropriate block indices, we assign the line ids
            for word in cermine_data[page_id].words:
                if word.block_id == -1:
                    word.block_id = word.line_id + PADDING_CONSTANT

            row_item = self.convert_token_data_to_json(cermine_data[page_id].words)

            if len(row_item["words"]) > 0:

                all_token_data.append(row_item)
                all_files.append(f"{sha}-{page_id}")

        return all_token_data, all_files

class S2VLAnnotationGeneratorWithVisionBox(S2VLAnnotationGenerator):

    def create_annotation_for_sha(self, sha):

        all_token_data = []
        all_files = []

        raw_blocks = self.raw_annotation.load_annotation_for_sha(sha)
        cermine_data = self.cermine_annotation.load_annotations_for_sha(sha)
        vision_data = self.vision_annotation.load_annotations_for_sha(sha)

        for page_id in cermine_data.keys():
            blocks = [
                b for b in raw_blocks[page_id] if b.type in self.selected_categories
            ]

            # Pass 1: O(n) Initialize ids and categories
            for word in cermine_data[page_id].words:
                word.line_id = -1
                word.block_id = -1
                word.type = self.default_category

            # Pass 2: O(mn) Assign token categories
            for word in cermine_data[page_id].words:
                for block in blocks:
                    if word.is_in(block, center=True):
                        word.type = block.type

            # Pass 3: O(mn) Assign token block-category ids
            used_lines_for_assign_line_ids = cermine_data[page_id].lines
            used_blocks_for_assign_block_ids = vision_data[page_id].blocks
            for word in cermine_data[page_id].words:
                for _l in used_lines_for_assign_line_ids:
                    if word.is_in(_l, center=True):
                        word.line_id = _l.id

                for _b in used_blocks_for_assign_block_ids:
                    if word.is_in(_b, center=True):
                        word.block_id = _b.id

            # Pass 4: O(n) In case some blocks are not assigned with the
            # appropriate block indices, we assign the line ids
            for word in cermine_data[page_id].words:
                if word.block_id == -1:
                    word.block_id = word.line_id + PADDING_CONSTANT

            row_item = self.convert_token_data_to_json(cermine_data[page_id].words)

            if len(row_item["words"]) > 0:

                all_token_data.append(row_item)
                all_files.append(f"{sha}-{page_id}")

        return all_token_data, all_files

class S2VLAnnotationGeneratorWithVisionLine(S2VLAnnotationGeneratorWithGTBox):
    
    def create_annotation_for_sha(self, sha):

        all_token_data = []
        all_files = []

        raw_blocks = self.raw_annotation.load_annotation_for_sha(sha)
        cermine_data = self.cermine_annotation.load_annotations_for_sha(sha)
        vision_data = self.vision_annotation.load_annotations_for_sha(sha)

        for page_id in cermine_data.keys():
            blocks = [
                b for b in raw_blocks[page_id] if b.type in self.selected_categories
            ]
            blocks = self.order_blocks_based_on_token_ids(
                blocks, cermine_data[page_id].words
            )

            # Pass 1: O(n) Initialize ids and categories
            for word in cermine_data[page_id].words:
                word.line_id = -1
                word.block_id = -1
                word.type = self.default_category

            # Pass 2: O(mn) Assign token categories
            for word in cermine_data[page_id].words:
                for block in blocks:
                    if word.is_in(block, center=True):
                        word.type = block.type

            # Pass 3: O(mn) Assign token block-category ids
            used_lines_for_assign_line_ids = vision_data[page_id].lines
            used_blocks_for_assign_block_ids = vision_data[page_id].blocks

            for word in cermine_data[page_id].words:
                for _l in used_lines_for_assign_line_ids:
                    if word.is_in(_l, center=True):
                        word.line_id = _l.id

                for _b in used_blocks_for_assign_block_ids:
                    if word.is_in(_b, center=True):
                        word.block_id = _b.id

            # Pass 4: O(n) In case some blocks are not assigned with the
            # appropriate block indices, we assign the line ids
            for word in cermine_data[page_id].words:
                if word.block_id == -1:
                    word.block_id = word.line_id + PADDING_CONSTANT

            row_item = self.convert_token_data_to_json(cermine_data[page_id].words)

            if len(row_item["words"]) > 0:

                all_token_data.append(row_item)
                all_files.append(f"{sha}-{page_id}")

        return all_token_data, all_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation-folder", type=str, help="The path to the annotation folder"
    )
    parser.add_argument(
        "--annotation-table", type=str, help="The table with sha-annotator name"
    )
    parser.add_argument(
        "--cermine-pdf-dir",
        type=str,
        help="The path to the folder containing the PDF and CERMINED results",
    )
    parser.add_argument(
        "--cermine-csv-dir",
        type=str,
        help="The path to the folder with CERMINED results stored in csv",
    )
    parser.add_argument(
        "--vision-csv-dir",
        type=str,
        help="The path to the folder with VISION Model results stored in csv",
    )
    parser.add_argument(
        "--export-folder", type=str, help="The folder for storing the data"
    )
    parser.add_argument("--config", type=str, help="The path to the config file")

    parser.add_argument("--use-gt-block", action="store_true")
    parser.add_argument("--use-vision-box", action="store_true")
    parser.add_argument("--use-vision-line", action="store_true")
    parser.add_argument("--few-shot-mutual-test-set", action="store_true")
    parser.add_argument("--few-shot-cv", action="store_true")

    args = parser.parse_args()

    raw_annotation = RawAnnotation(args.annotation_table, args.annotation_folder)
    cermine_annotation = CERMINEAnnotation(args.cermine_pdf_dir, args.cermine_csv_dir)
    vision_annotation = VISIONAnnotation(None, args.vision_csv_dir)

    config = load_json(args.config)

    if args.use_gt_block:
        s2vl = S2VLAnnotationGeneratorWithGTBox(
            args.annotation_table,
            raw_annotation,
            cermine_annotation,
            config["selected_categories"],
            config["default_category"],
        )
        save_folder = f"{args.export_folder}-gtbox"
    elif args.use_vision_box:
        s2vl = S2VLAnnotationGeneratorWithVisionBox(
            args.annotation_table,
            raw_annotation,
            cermine_annotation,
            config["selected_categories"],
            config["default_category"],
            vision_annotation=vision_annotation,
        )
        save_folder = f"{args.export_folder}-visionbox"
    elif args.use_vision_line:
        s2vl = S2VLAnnotationGeneratorWithVisionLine(
            args.annotation_table,
            raw_annotation,
            cermine_annotation,
            config["selected_categories"],
            config["default_category"],
            vision_annotation=vision_annotation,
        )
        save_folder = f"{args.export_folder}-visionline-v2"
    else:
        s2vl = S2VLAnnotationGenerator(
            args.annotation_table,
            raw_annotation,
            cermine_annotation,
            config["selected_categories"],
            config["default_category"],
        )
        save_folder = args.export_folder

    s2vl.create_annotations()
    s2vl.save_annotation_cv(f"{save_folder}-cv", 5)
    # if args.few_shot_mutual_test_set:
