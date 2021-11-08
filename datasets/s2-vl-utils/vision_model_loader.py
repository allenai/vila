from genericpath import exists
from glob import glob
import os
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
import layoutparser as lp
from pdf2image import convert_from_path

from vision_postprocessor import pipeline, create_structure_df

class S2VLLoader:
    def __init__(self, pdf_path):

        self.pdf_path = pdf_path
        self.all_pdfs = glob(f"{self.pdf_path}/*.pdf")

    def __getitem__(self, idx):
        assert idx < len(self.all_pdfs)
        return self.load_sample(self.all_pdfs[idx])

    def load_sample(self, pdf_path):
        return pdf_path, convert_from_path(pdf_path, dpi=72)

    def __len__(self):
        return len(self.all_pdfs)


def calculate_overlapping_coefficient(box1, box2):
    x1, y1, x2, y2 = box1.coordinates
    a1, b1, a2, b2 = box2.coordinates

    if x2 < a1 or x1 > a2 or y1 > b2 or y2 < b1:  # Bottom or top
        return 0

    else:
        intersection = lp.Rectangle(
            x_1=max(x1, a1), y_1=max(y1, b1), x_2=min(x2, a2), y_2=min(y2, b2)
        )
        return intersection.area / min(box1.area, box2.area)


THRESHOLD_FOR_OVERLAPPING_BLOCKS = 0.5


def filter_out_non_overlapping_block(blocks):

    boxes_to_remove = np.zeros(len(blocks))
    for box2 in blocks:
        for box1 in blocks:

            if box1.id == box2.id:
                continue

            if boxes_to_remove[box1.id] or boxes_to_remove[box2.id]:
                continue

            if (
                calculate_overlapping_coefficient(box1, box2)
                > THRESHOLD_FOR_OVERLAPPING_BLOCKS
            ):
                if box1.area >= box2.area:
                    boxes_to_remove[box2.id] = 1
                else:
                    boxes_to_remove[box1.id] = 1

    return [b for b in blocks if not boxes_to_remove[b.id]]


def convert_blocks_to_df(blocks_line):
    blocks_to_save = [[ele.id, *ele.coordinates, ele.type, ele.score] for ele in blocks_line]

    df = pd.DataFrame(
        blocks_to_save,
        columns=[
            "id",
            "x_1",
            "y_1",
            "x_2",
            "y_2",
            "category",
            "confidence",
        ],
    )

    df[["x_1", "y_1", "x_2", "y_2"]] = df[["x_1", "y_1", "x_2", "y_2"]].astype("int")
    return df


def textline_detection(base_path):
    def detect_lines_for_image(pdf_image):
        blocks_line = model_line.detect(pdf_image)
        blocks_line = [token.set(id=idx) for idx, token in enumerate(blocks_line)]
        blocks_line = filter_out_non_overlapping_block(blocks_line)
        return blocks_line

    model_line = lp.Detectron2LayoutModel(
        config_path=f"https://www.dropbox.com/s/hd21tarnhbj1p1o/config.yaml?dl=1", # This is the line detection model trained using the GROTOAP2 dataset
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.35,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.8,
        ],
        label_map={0: "line"},
    )

    loader = S2VLLoader(f"{base_path}/pdfs")

    for pdf_path in tqdm(loader.all_pdfs):
        pdf_name = pdf_path.split("/")[-1].replace(".pdf", "")
        pdf_images = convert_from_path(pdf_path, dpi=72)
        for pid, pdf_image in enumerate(pdf_images):

            blocks_line = detect_lines_for_image(pdf_image)

            df = convert_blocks_to_df(blocks_line)

            if len(pdf_images) == 1 and len(pdf_name.split("-")) == 2:
                df.to_csv(f"{base_path}/lines/{pdf_name}.csv", index=None)
            else:
                df.to_csv(f"{base_path}/lines/{pdf_name}-{pid:02d}.csv", index=None)


def textblock_detection(base_path):
    def detect_blocks_for_image(pdf_image):
        blocks1 = block_predictorA.detect(pdf_image)
        blocks2 = block_predictorB.detect(pdf_image)

        blocks = blocks1 + blocks2
        blocks = sorted(blocks, key=lambda ele: ele.coordinates[1])
        blocks = [token.set(id=idx) for idx, token in enumerate(blocks)]
        blocks = filter_out_non_overlapping_block(blocks)
        return blocks

    block_predictorA = lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.50,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.4,
        ],
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
    )

    block_predictorB = lp.Detectron2LayoutModel(
        config_path="lp://MFD/faster_rcnn_R_50_FPN_3x/config",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.6,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.2,
        ],
        label_map={1: "equation"},
    )

    loader = S2VLLoader(f"{base_path}/pdfs")

    for pdf_path in tqdm(loader.all_pdfs):
        pdf_name = pdf_path.split("/")[-1].replace(".pdf", "")
        pdf_images = convert_from_path(pdf_path, dpi=72)

        for pid, pdf_image in enumerate(pdf_images):
            blocks = detect_blocks_for_image(pdf_image)
            df = convert_blocks_to_df(blocks)

            if len(pdf_images) == 1 and len(pdf_name.split("-")) == 2:
                df.to_csv(f"{base_path}/blocks/{pdf_name}.csv", index=None)
            else:
                df.to_csv(f"{base_path}/blocks/{pdf_name}-{pid:02d}.csv", index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, help="The path to the source files of a dataset, e.g., sources/s2-vl-ver1")
    args = parser.parse_args()

    os.makedirs(f"{args.base_path}/blocks", exist_ok=True)

    if len(glob(f"{args.base_path}/blocks/*.csv")) > 0:
        print("Text Blocks already detected")
    else:
        print("Running Text Block Detection")
        textblock_detection(args.base_path)

    os.makedirs(f"{args.base_path}/lines", exist_ok=True)
    if len(glob(f"{args.base_path}/lines/*.csv")) > 0:
        print("Text Lines already detected")
    else:
        print("Running Text Line Detection")
        textline_detection(args.base_path)
        
    target_dir = f"{args.base_path}/condensed"
    os.makedirs(target_dir, exist_ok=True)
    for filename in tqdm(glob(f"{args.base_path}/pdfs/*.pdf")):
        res = os.path.basename(filename).split(".")[0].split("-")
        if len(res)==2:
            pdf_sha, pid = res
            blocks, lines, tokens, additional_blocks, additional_lines = pipeline(args.base_path, pdf_sha, pid)
            df = create_structure_df(tokens, blocks, lines)
            df.to_csv(f"{target_dir}/{pdf_sha}-{pid}.csv", index=None)
        else:
            pdf_sha = res[0]
            pids = len(glob(f"{args.base_path}/pdfs/{pdf_sha}-*.csv"))
            for pid in range(pids):
                blocks, lines, tokens, additional_blocks, additional_lines = pipeline(args.base_path, pdf_sha, f"{pid:02d}")
                df = create_structure_df(tokens, blocks, lines)
                df.to_csv(f"{target_dir}/{pdf_sha}-{pid:02d}.csv", index=None)