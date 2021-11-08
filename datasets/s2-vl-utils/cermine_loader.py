from typing import List, Union, Dict, Any, Tuple
from dataclasses import dataclass
from glob import glob
import os
import subprocess


from tqdm import tqdm
from bs4 import BeautifulSoup
import layoutparser as lp
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class PageData:
    blocks: List[lp.TextBlock]
    lines: List[lp.TextBlock]
    words: List[lp.TextBlock]

    def to_dataframe(
        self,
        keep_token_index=True,
        export_font=False,
        normalize_coordinates=False,
        canvas_width=None,
        canvas_height=None,
    ) -> pd.DataFrame:

        if not export_font:
            blocks_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    ele.type,
                    -1,
                    -1,
                    True,
                    False,
                ]
                for ele in self.blocks
            ]
            lines_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    ele.type,
                    ele.parent,
                    -1,
                    False,
                    True,
                ]
                for ele in self.lines
            ]
            parent_block_id_for_line_id = {ele.id: ele.parent for ele in self.lines}
            tokens_to_save = [
                [
                    ele.id if keep_token_index else idx,
                    *ele.coordinates,
                    ele.text,
                    ele.type,
                    parent_block_id_for_line_id[ele.parent],  # Cvt to block-level id
                    ele.parent,
                    False,
                    False,
                ]
                for idx, ele in enumerate(self.words, start=len(blocks_to_save))
            ]
            df = pd.DataFrame(
                blocks_to_save + lines_to_save + tokens_to_save,
                columns=[
                    "id",
                    "x_1",
                    "y_1",
                    "x_2",
                    "y_2",
                    "text",
                    "category",
                    "block_id",
                    "line_id",
                    "is_block",
                    "is_line",
                ],
            )
        else:
            blocks_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    None,
                    ele.type,
                    -1,
                    -1,
                    True,
                    False,
                ]
                for ele in self.blocks
            ]
            lines_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    None,
                    ele.type,
                    ele.parent,
                    -1,
                    False,
                    True,
                ]
                for ele in self.lines
            ]
            parent_block_id_for_line_id = {ele.id: ele.parent for ele in self.lines}
            tokens_to_save = [
                [
                    ele.id if keep_token_index else idx,
                    *ele.coordinates,
                    ele.text,
                    ele.font,
                    ele.type,
                    parent_block_id_for_line_id[ele.parent],  # Cvt to block-level id
                    ele.parent,
                    False,
                    False,
                ]
                for idx, ele in enumerate(self.words, start=len(blocks_to_save))
            ]
            df = pd.DataFrame(
                blocks_to_save + lines_to_save + tokens_to_save,
                columns=[
                    "id",
                    "x_1",
                    "y_1",
                    "x_2",
                    "y_2",
                    "text",
                    "font",
                    "category",
                    "block_id",
                    "line_id",
                    "is_block",
                    "is_line",
                ],
            )

        if normalize_coordinates:
            assert canvas_width is not None
            assert canvas_height is not None
            df[["x_1", "x_2"]] = (df[["x_1", "x_2"]] / canvas_width * 1000).astype(
                "int"
            )
            df[["y_1", "y_2"]] = (df[["y_1", "y_2"]] / canvas_height * 1000).astype(
                "int"
            )

        return df


class GrotoapDataset:
    def __init__(self, base_dir: str, dataset_folder_name: str = "dataset"):

        self.base_dir = base_dir
        self.dataset_folder_name = dataset_folder_name
        self.all_xml_files = glob(
            f"{self.base_dir}/{self.dataset_folder_name}/*/*.cxml"
        )

    def load_xml(self, xml_filename: str):
        with open(xml_filename, "r") as fp:
            soup = BeautifulSoup(fp, "lxml")

        pages = soup.find_all("page")

        parsed_page_data = {
            idx: self.parse_page_xml(page) for idx, page in enumerate(pages)
        }

        return parsed_page_data

    def parse_page_xml(self, page: "bs4.element.Tag") -> PageData:

        blocks = []
        lines = []
        words = []

        word_id = 0
        line_id = 0
        all_zones = page.find_all("zone")
        if all_zones is None:
            return PageData()

        for zone_id, zone in enumerate(all_zones):

            words_in_this_block = []
            # Fetch the zone
            v1, v2 = zone.find("zonecorners").find_all("vertex")
            block_type = zone.find("classification").find("category")["value"]
            block = lp.TextBlock(
                lp.Rectangle(
                    float(v1["x"]), float(v1["y"]), float(v2["x"]), float(v2["y"])
                ),
                type=block_type,
                id=zone_id,
            )

            # Fetch lines
            all_lines = zone.find_all("line")
            if all_lines is None:
                continue

            for line in all_lines:

                words_in_this_line = []

                v1, v2 = line.find("linecorners").find_all("vertex")
                current_line = lp.TextBlock(
                    lp.Rectangle(
                        float(v1["x"]),
                        float(v1["y"]),
                        float(v2["x"]),
                        float(v2["y"]),
                    ),
                    type=block_type,
                    parent=zone_id,
                    id=line_id,
                )

                # Fetch words
                all_words = line.find_all("word")
                if all_words is None:
                    continue

                for word in line.find_all("word"):
                    v1, v2 = word.find("wordcorners").find_all("vertex")
                    words_in_this_line.append(
                        lp.TextBlock(
                            lp.Rectangle(
                                float(v1["x"]),
                                float(v1["y"]),
                                float(v2["x"]),
                                float(v2["y"]),
                            ),
                            type=block_type,
                            text="".join(
                                [ele["value"] for ele in word.find_all("gt_text")]
                            ),
                            id=word_id,
                            parent=line_id,
                        )
                    )
                    word_id += 1

                current_line.text = " ".join(ele.text for ele in words_in_this_line)
                line_id += 1
                words_in_this_block.extend(words_in_this_line)
                lines.append(current_line)

            block.text = " ".join(ele.text for ele in words_in_this_block)
            blocks.append(block)
            words.extend(words_in_this_block)

        return PageData(blocks, lines, words)

    def convert_xml_to_page_token(self, xml_filename, export_path):

        savename = "-".join(xml_filename.split("/")[-2:]).rstrip(".cxml")
        parsed_page_data = self.load_xml(xml_filename)
        print(f"Processing {savename}")
        for page_id, page_data in parsed_page_data.items():

            if os.path.exists(f"{export_path}/{savename}-{page_id}.csv"):
                continue

            df = page_data.to_dataframe()
            df.to_csv(f"{export_path}/{savename}-{page_id}.csv", index=None)

    def convert_to_page_token_table(self, export_path: str, n_jobs=20):

        if not os.path.exists(export_path):
            os.makedirs(export_path)
            print(f"Creating the export directory {export_path}")
        else:
            print(f"Overwriting existing exports in {export_path}")

        Parallel(n_jobs=n_jobs)(
            delayed(self.convert_xml_to_page_token)(xml_filename, export_path)
            for xml_filename in tqdm(self.all_xml_files)
        )


class CERMINELoader(GrotoapDataset):
    def __init__(self):
        pass

    @staticmethod
    def corner_to_rectangle(corners):
        corners = corners.find_all("vertex")
        corners = np.array([(float(ele["x"]), float(ele["y"])) for ele in corners])
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        return lp.Rectangle(x1, y1, x2, y2)

    def parse_page_xml(self, page: "bs4.element.Tag") -> PageData:

        blocks = []
        lines = []
        words = []

        word_id = 0
        line_id = 0
        all_zones = page.find_all("zone")
        if all_zones is None:
            return PageData()

        for zone_id, zone in enumerate(all_zones):

            words_in_this_block = []
            # Fetch the zone
            rect = self.corner_to_rectangle(zone.find("zonecorners"))
            block_type = zone.find("classification").find("category")["value"]
            block = lp.TextBlock(
                rect,
                type=block_type,
                id=zone_id,
            )

            # Fetch lines
            all_lines = zone.find_all("line")
            if all_lines is None:
                continue

            for line in all_lines:

                words_in_this_line = []

                rect = self.corner_to_rectangle(line.find("linecorners"))
                current_line = lp.TextBlock(
                    rect,
                    type=block_type,
                    parent=zone_id,
                    id=line_id,
                )

                # Fetch words
                all_words = line.find_all("word")
                if all_words is None:
                    continue

                for word in line.find_all("word"):
                    rect = self.corner_to_rectangle(word.find("wordcorners"))
                    words_in_this_line.append(
                        lp.TextBlock(
                            rect,
                            type=block_type,
                            text="".join(
                                [ele["value"] for ele in word.find_all("gt_text")]
                            ),
                            id=word_id,
                            parent=line_id,
                        )
                    )
                    word_id += 1

                current_line.text = " ".join(ele.text for ele in words_in_this_line)
                line_id += 1
                words_in_this_block.extend(words_in_this_line)
                lines.append(current_line)

            block.text = " ".join(ele.text for ele in words_in_this_block)
            blocks.append(block)
            words.extend(words_in_this_block)

        return PageData(blocks, lines, words)


CERMINE_LOADER = CERMINELoader()


def process_cermine_annotation(sha, pdf_path, token_path):
    filename = f"{pdf_path}/{sha}.cxml"

    try:
        xml_data = CERMINE_LOADER.load_xml(filename)
    except:
        print("error CERMINE parsing for ", sha)
        return None

    # _, pdf_images = pdf_extractor.load_tokens_and_image(filename.replace('.cxml', '.pdf'), resize_image=True)

    # if len(xml_data) != len(pdf_images):
    #     print("error CERMINE parsing for ", sha)
    #     return None

    if (
        len(xml_data) == 1 and len(sha.split("-")) == 2
    ):  # it is a single page pdf for an individual page
        xml_data[0].to_dataframe().to_csv(f"{token_path}/{sha}.csv", index=None)
    else:
        for page_id in range(len(xml_data)):
            xml_data[page_id].to_dataframe().to_csv(
                f"{token_path}/{sha}-{page_id:02d}.csv", index=None
            )

def get_file_sha(filename):
    return filename.split("/")[-1].split(".")[0]

# parse the arguments of base_path and run process_cermine_annotation
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, help="The path to the source files of a dataset, e.g., sources/s2-vl-ver1")
    parser.add_argument("--cermine-path", type=str)
    parser.add_argument("--njobs", type=int, default=2)
    args = parser.parse_args()

    # folder structure 
    base_path = args.base_path
    pdf_path = f"{base_path}/pdfs"
    token_path = f"{base_path}/tokens"

    # verify the existence of the files 
    if not os.path.exists(pdf_path) or len(glob(f"{pdf_path}/*.pdf")) == 0:
        print(f"The PDF path {pdf_path} does not exist! Please try download the dataset first.")
        exit()

    # Run cermine parsing 
    if len(glob(f"{pdf_path}/*.cxml")) == 0:
        CERMINE_IMP_NAME = "cermine-impl-1.13-jar-with-dependencies.jar"
        cermine_imp_name = args.cermine_path if os.path.exists(args.cermine_path) else CERMINE_IMP_NAME
        cermine_prog_name = "pl.edu.icm.cermine.PdfBxStructureExtractor"
        subprocess.call(
            ["java", "-cp", cermine_imp_name, cermine_prog_name, "-path", pdf_path]
        )
        print("Finish processing all the PDFs using CERMINE")
    else:
        print("CERMINE XML files already exist")

    # process the cermine files
    if not os.path.exists(token_path):
        os.makedirs(token_path)
        
    Parallel(n_jobs=args.njobs)(
        delayed(process_cermine_annotation)(get_file_sha(filename), pdf_path, token_path)
        for filename in tqdm(glob(f"{pdf_path}/*.pdf"))
    )
    print("Finish converting all the CERMINE XMLS to csv")