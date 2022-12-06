import itertools
import re
import os
import argparse
from pathlib import Path
from functools import reduce
from typing import List, Tuple, Dict
from collections import Counter
from functools import partial

from PIL import Image
import layoutparser as lp  # For visualization
import pandas as pd
import numpy as np

from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor

FLOATING_REGION_TYPES = ["Table", "Figure", "Equation"]
MAX_POSSIBLE_DISTANCES_CAPTION = 375


def convert_sequence_tagging_to_spans(
    token_prediction_sequence: lp.Layout, key
) -> List[Tuple[int, int, int]]:
    """For a sequence of token predictions, convert them to spans
    of consecutive same predictions.

    Args:
        token_prediction_sequence (lp.Layout)
    Returns:
        List[Tuple[int, int, int]]: A list of (start, end, label)
            of consecutive prediction of the same label.
    """

    prev_len = 0
    spans = []

    for gp, seq in itertools.groupby(token_prediction_sequence, key=key):
        cur_len = len(list(seq))
        yield (prev_len, prev_len + cur_len, gp)
        prev_len = prev_len + cur_len


def aggregate_consecutive_group_intervals(gp):
    return pd.Series(
        {
            "page": gp["page"].iloc[0],
            "type": gp["type"].iloc[0],
            "intervals": list(zip(gp["start"], gp["end"])),
        }
    )


def union(block1, block2):
    x11, y11, x12, y12 = block1.coordinates
    x21, y21, x22, y22 = block2.coordinates

    block = lp.Rectangle(min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))
    if isinstance(block1, lp.TextBlock):
        return lp.TextBlock(
            block,
            id=block1.id,
            type=block1.type,
            text=block1.text + " " + block2.text,
            parent=block1.parent,
            next=block1.next,
        )
    else:
        return block


def union_blocks(blocks):
    return reduce(union, blocks)


def union_intervals(intervals: List[Tuple[int, int]]) -> Tuple[int, int]:
    return (min(e[0] for e in intervals), max(e[1] for e in intervals))


def select_tokens_based_on_intervals(tokens, intervals):
    return lp.Layout(
        list(itertools.chain.from_iterable([tokens[slice(*ele)] for ele in intervals]))
    )


def calculate_block_distance(block_A, block_B):
    # just the manhattan distance

    center_A = block_A.block.center
    center_B = block_B.block.center
    return sum(abs(a - b) for a, b in zip(center_A, center_B))


def calculate_pairwise_distance(blocks_A, blocks_B):

    return np.array(
        [
            [calculate_block_distance(box1, box2) for box2 in blocks_B]
            for box1 in blocks_A
        ]
    )


def pair_figure_caption_blocks(figure_blocks, caption_blocks):

    if len(figure_blocks) == 0:
        return [], [], list(range(len(caption_blocks)))
    if len(caption_blocks) == 0:
        return [], list(range(len(figure_blocks))), []

    distances = calculate_pairwise_distance(figure_blocks, caption_blocks)
    best_matching_caption_block_ids = distances.argmin(axis=1)

    paired_fig_cap = []
    used_figures = []
    used_captions = []

    for fig_id, cap_id in enumerate(best_matching_caption_block_ids):

        if cap_id not in used_captions:

            if distances[fig_id, cap_id] > MAX_POSSIBLE_DISTANCES_CAPTION:
                print("Unreasonable figure caption distances")
                continue

            paired_fig_cap.append((fig_id, cap_id))
            used_figures.append(fig_id)
            used_captions.append(cap_id)

    unused_figures = [i for i in range(len(figure_blocks)) if i not in used_figures]
    unused_captions = [i for i in range(len(caption_blocks)) if i not in used_captions]

    return paired_fig_cap, unused_figures, unused_captions


def get_caption_header(caption):
    re_fig = re.search(r"figure \d", caption.lower())
    if re_fig:
        return re_fig.group(0)
    re_table = re.search(r"table \d", caption.lower())
    if re_table:
        return re_table.group(0)
    return None


def get_text_coord_for_intervals(row, page_tokens):
    page_id = row["page"]
    tokens = page_tokens[page_id].tokens
    cur_block = union_blocks(select_tokens_based_on_intervals(tokens, row["intervals"]))
    return pd.Series(
        [cur_block.text, *cur_block.coordinates], index=["text", "x1", "y1", "x2", "y2"]
    )


def pipeline(
    *,
    input_pdf: Path = None,
    output_path: Path = None,
    pdf_extractor=None,
    vision_model1=None,
    vision_model2=None,
    pdf_predictor=None,
    relative_coordinates=False,
    return_csv=False,
):
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(input_pdf)

    for idx, page_token in enumerate(page_tokens):

        blocks = vision_model1.detect(page_images[idx]) + vision_model2.detect(
            page_images[idx]
        )
        for idx, b in enumerate(blocks):
            b.set(id=idx, inplace=True)
        page_token.blocks = blocks

    all_preds = []
    for idx, page_token in enumerate(page_tokens):
        predicted_types = pdf_predictor.predict_page(page_token, return_type="list")

        for (start, end, _) in convert_sequence_tagging_to_spans(
            page_token.tokens, lambda ele: ele.line_id
        ):
            unique_preds = Counter(predicted_types[start:end])

            # Per line majority voting
            if len(unique_preds) > 1:
                # print(predicted_types[start:end])
                _type = unique_preds.most_common()[0][0]
            else:
                _type = predicted_types[start]

            all_preds.append([idx, _type, start, end])

    df = pd.DataFrame(all_preds, columns=["page", "type", "start", "end"])
    df = (
        df.groupby(
            [((df.type != df.type.shift()) | (df.page != df.page.shift())).cumsum()]
        )
        .apply(aggregate_consecutive_group_intervals)
        .reset_index()
    )

    final_captions = []

    for page_id, page_captions in df[df["type"] == "Caption"].groupby("page"):

        cur_page = page_tokens[page_id]

        figure_table_blocks = [
            block for block in cur_page.blocks if block.type in ["Figure", "Table"]
        ]
        caption_blocks = page_captions.apply(
            lambda e: union_blocks(
                cur_page.tokens[slice(*union_intervals(e["intervals"]))]
            ).set(id=e["index"]),
            axis=1,
        ).tolist()

        paired_fig_cap, unused_figures, unused_captions = pair_figure_caption_blocks(
            figure_table_blocks, caption_blocks
        )

        for ele in paired_fig_cap:
            final_captions.append(
                [
                    caption_blocks[ele[1]].id,
                    figure_table_blocks[ele[0]].type,
                    page_id,
                    figure_table_blocks[ele[0]].id,
                    caption_blocks[ele[1]].text,
                    *caption_blocks[ele[1]].coordinates,
                ]
            )

    final_equations = []

    for page_id, page_equations in df[df["type"] == "Equation"].groupby("page"):
        cur_page = page_tokens[page_id]

        visual_equation_blocks = [
            block for block in cur_page.blocks if block.type in ["Equation"]
        ]
        equation_blocks = page_equations.apply(
            lambda e: union_blocks(
                cur_page.tokens[slice(*union_intervals(e["intervals"]))]
            ).set(id=e["index"]),
            axis=1,
        ).tolist()

        paired_equations, unused_figures, unused_captions = pair_figure_caption_blocks(
            visual_equation_blocks, equation_blocks
        )

        for ele in paired_equations:
            final_equations.append(
                [
                    equation_blocks[ele[1]].id,
                    visual_equation_blocks[ele[0]].type,
                    page_id,
                    visual_equation_blocks[ele[0]].id,
                    equation_blocks[ele[1]].text,
                    *equation_blocks[ele[1]].coordinates,
                ]
            )
    final_captions += final_equations
    # Because the dealing of equations are almost identical as the figure
    # blocks, we can merge the list.

    final_section_headers = []

    for page_id, page_sections in df[df["type"] == "Section"].groupby("page"):
        cur_page = page_tokens[page_id]
        for idx, page_section in page_sections.iterrows():
            # for interval in page_section["intervals"]:
            # print(cur_page.tokens[slice(*interval)].get_texts())
            # print([cur_page.tokens[idx].font for idx in interval])
            # print(Counter([cur_page.tokens[idx].font for idx in interval]).most_common()[0][0])
            section_header_fonts = [
                Counter([cur_page.tokens[idx].font for idx in interval]).most_common()[
                    0
                ][0]
                for interval in page_section["intervals"]
            ]

            prev_len = 0
            for gp, seq in itertools.groupby(section_header_fonts):
                cur_len = prev_len + len(list(seq))
                cur_interval = union_intervals(
                    page_section["intervals"][prev_len:cur_len]
                )
                cur_block = union_blocks(cur_page.tokens[slice(*cur_interval)])
                final_section_headers.append(
                    [
                        page_section["index"],
                        page_id,
                        cur_interval,
                        cur_block.text,
                        *cur_block.coordinates,
                    ]
                )
                prev_len = cur_len

    final_paragraphs = []

    for page_id, page_paragraphs in df[df["type"] == "Paragraph"].groupby("page"):
        cur_page = page_tokens[page_id]
        for idx, page_paragraph in page_paragraphs.iterrows():
            interval_lines = [
                cur_page.lines[cur_page.tokens[int[0]].line_id]
                for int in page_paragraph["intervals"]
            ]

            line_to_block_id = []
            for line in interval_lines:
                found = False
                for block in cur_page.blocks:
                    if line.is_in(block, center=True):
                        line_to_block_id.append(block.id)
                        found = True
                        break
                if not found:
                    line_to_block_id.append(None)

            assert len(line_to_block_id) == len(
                interval_lines
            ), f"{len(line_to_block_id)}, {len(interval_lines)}"
            prev_len = 0
            # print(line_to_block_id)
            for gp, seq in itertools.groupby(line_to_block_id):
                cur_len = prev_len + len(list(seq))
                cur_intervals = page_paragraph["intervals"][prev_len:cur_len]
                cur_block = union_blocks(
                    select_tokens_based_on_intervals(cur_page.tokens, cur_intervals)
                )
                final_paragraphs.append(
                    [
                        page_paragraph["index"],
                        page_id,
                        cur_intervals,
                        cur_block.text,
                        *cur_block.coordinates,
                    ]
                )
                prev_len = cur_len

    caption_df = pd.DataFrame(
        final_captions,
        columns=[
            "index",
            "block_type",
            "page",
            "block_id",
            "text",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )
    section_df = pd.DataFrame(
        final_section_headers,
        columns=["index", "page", "intervals", "text", "x1", "y1", "x2", "y2"],
    )
    paragraph_df = pd.DataFrame(
        final_paragraphs,
        columns=["index", "page", "intervals", "text", "x1", "y1", "x2", "y2"],
    )

    tmp = df.copy()
    tmp["text"] = None

    tmp = tmp.merge(
        paragraph_df, on=["index", "page"], how="outer", suffixes=["", "_paragraph"]
    )
    idx = tmp[~tmp["intervals_paragraph"].isna()].index
    tmp.loc[idx, "intervals"] = tmp.loc[idx, "intervals_paragraph"]
    tmp.loc[idx, "text"] = tmp.loc[idx, "text_paragraph"]
    tmp = tmp.drop(columns=["intervals_paragraph", "text_paragraph"])

    tmp = tmp.merge(
        section_df, on=["index", "page"], how="outer", suffixes=["", "_section"]
    )
    idx = tmp[~tmp["intervals_section"].isna()].index
    columns_to_merge = ["intervals", "text", "x1", "y1", "x2", "y2"]
    for col in columns_to_merge:
        tmp.loc[idx, col] = tmp.loc[idx, f"{col}_section"]
    tmp = tmp.drop(columns=[f"{col}_section" for col in columns_to_merge])

    tmp = tmp.merge(
        caption_df, on=["index", "page"], how="outer", suffixes=["", "_caption"]
    )
    idx = tmp[~tmp["text_caption"].isna()].index
    columns_to_merge = ["text", "x1", "y1", "x2", "y2"]
    for col in columns_to_merge:
        tmp.loc[idx, col] = tmp.loc[idx, f"{col}_caption"]
    tmp = tmp.drop(columns=[f"{col}_caption" for col in columns_to_merge])

    idx = tmp[tmp["text"].isna()].index
    tmp.loc[idx, ["text", "x1", "y1", "x2", "y2"]] = tmp.loc[idx].apply(
        partial(get_text_coord_for_intervals, page_tokens=page_tokens), axis=1
    )

    if relative_coordinates:
        for page_id in tmp["page"].unique():
            cur_page_w, cur_page_h = page_images[page_id].size
            idx = tmp[tmp["page"] == page_id].index
            tmp.loc[idx, "x1"] /= cur_page_w
            tmp.loc[idx, "x2"] /= cur_page_w
            tmp.loc[idx, "y1"] /= cur_page_h
            tmp.loc[idx, "y2"] /= cur_page_h

    if return_csv:
        return tmp.drop(columns=["index", "intervals"])

    save_path = output_path / input_pdf.stem
    os.makedirs(save_path, exist_ok=True)

    # Save structure
    tmp.drop(columns=["index", "intervals"]).to_csv(save_path / "structure.csv")

    # Save figures and tables
    os.makedirs(save_path / "figures", exist_ok=True)
    for pid, page_token in enumerate(page_tokens):
        page_image = np.array(page_images[pid])
        for block in page_token.blocks:
            if block.type not in FLOATING_REGION_TYPES:
                continue
            figure_screenshot = block.pad(left=5, right=5, top=5, bottom=5).crop_image(
                page_image
            )
            figure_screenshot = Image.fromarray(figure_screenshot)
            figure_screenshot.save(save_path / f"figures/{pid:02d}-{block.id:02d}.png")


def build_predictors():
    pdf_extractor = PDFExtractor("pdfplumber")
    vision_model1 = lp.EfficientDetLayoutModel("lp://PubLayNet")
    vision_model2 = lp.EfficientDetLayoutModel("lp://MFD")
    pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained(
        "allenai/ivila-row-layoutlm-finetuned-s2vl-v2"
    )
    return pdf_extractor, vision_model1, vision_model2, pdf_predictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdf", type=str, default="default", help="desc")
    parser.add_argument("--output_path", type=str, default=".", help="desc")
    args = parser.parse_args()

    pdf_extractor, vision_model1, vision_model2, pdf_predictor = build_predictors()

    pipeline(
        input_pdf=Path(args.input_pdf),
        output_path=Path(args.output_path),
        pdf_extractor=pdf_extractor,
        vision_model1=vision_model1,
        vision_model2=vision_model2,
        pdf_predictor=pdf_predictor,
    )
