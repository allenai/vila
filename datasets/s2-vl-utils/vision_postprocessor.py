import math
from functools import partial
import json
import re
import random
from itertools import groupby
from collections import Counter, defaultdict
from copy import copy

import os
from PIL import Image
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from glob import glob
import pandas as pd
from scipy.sparse.csgraph import connected_components

NON_TEXTUAL_TYPES = ["table", "figure", "equation"]

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def get_most_common_element(lst):
    return Counter(lst).most_common(1)[0][0]

def get_most_common_token_type(tokens):
    return get_most_common_element([ele.type for ele in tokens])

def union_box(blocks):
    if len(blocks) == 0:
        # print("Warning: The length of blocks is 0!")
        rect = lp.Rectangle(0, 0, 0, 0)
        return lp.TextBlock(rect)
    else:
        x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")
        for block in blocks:
            bbox = block.coordinates
            x1 = min(x1, bbox[0])
            y1 = min(y1, bbox[1])
            x2 = max(x2, bbox[2])
            y2 = max(y2, bbox[3])
        rect = lp.Rectangle(int(x1), int(y1), int(x2), int(y2))
        return lp.TextBlock(rect, type=blocks[0].type)
    
def is_in(block_a, block_b, metric="center"):
    """A rewrite of the lp.LayoutElement.is_in function.
    We will use a soft_margin and center function by default.
    """
    if metric == "center":
        return block_a.is_in(
            block_b,
            soft_margin={"top": 1, "bottom": 1, "left": 1, "right": 1},
            center=True,
        )
    elif metric == "coef":
        return (
            calculate_overlapping_coefficient(block_a, block_b)
            > MIN_OVERLAPPING_THRESHOLD
        )
    elif metric == "any":
        return is_in(block_a, block_b, metric="center") or is_in(
            block_a, block_b, metric="coef"
        )
    
def is_non_textual_type(block):
    if isinstance(block.type, str):
        return block.type in NON_TEXTUAL_TYPES
    else:
        raise ValueError(f"strange block type data type {type(block.type)}")

def cvt_cermine_df_to_layout(row):

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

def cvt_line_df_to_layout(row):

    return lp.TextBlock(
        lp.Rectangle(
            row["x_1"],
            row["y_1"],
            row["x_2"],
            row["y_2"],
        ),
        id=row["id"],
    )

def cvt_block_df_to_layout(row):

    return lp.TextBlock(
        lp.Rectangle(
            row["x_1"],
            row["y_1"],
            row["x_2"],
            row["y_2"],
        ),
        id=row["id"],
        type=row["category"],
    )

def load_cermine_data_from_csv(filename):
    df = pd.read_csv(filename)
    if len(df) == 0:
        return None

    df = df[~df.text.isna()]
    if len(df) == 0:
        return None

    tokens_df = df[~df.is_line & ~df.is_block]

    return lp.Layout(tokens_df.apply(cvt_cermine_df_to_layout, axis=1).tolist())

def load_line_data_from_csv(filename):
    df = pd.read_csv(filename)
    return lp.Layout(
        df.apply(cvt_line_df_to_layout, axis=1).tolist()
    )

def load_block_data_from_csv(filename):
    df = pd.read_csv(filename)
    return lp.Layout(
        df.apply(cvt_block_df_to_layout, axis=1).tolist()
    )

def calculate_overlapping_coefficient(box1, box2):
    x1, y1, x2, y2 = box1.coordinates
    a1, b1, a2, b2 = box2.coordinates

    if x2 < a1 or x1 > a2 or y1 > b2 or y2 < b1:  # Bottom or top
        return 0

    min_area = min(box1.area, box2.area)
    if min_area == 0:
        return 0
    else:
        intersection = lp.Rectangle(
            x_1=max(x1, a1), y_1=max(y1, b1), x_2=min(x2, a2), y_2=min(y2, b2)
        )
        return intersection.area / min_area

def calculate_pairwise_overlapping_coefficient(blocks_A, blocks_B=None):
    if blocks_B is not None:
        return np.array(
            [
                [calculate_overlapping_coefficient(box1, box2) for box2 in blocks_B]
                for box1 in blocks_A
            ]
        )
    else:
        n = len(blocks_A)
        overlapping = np.zeros((n, n))
        for row in range(n):
            for col in range(row + 1, n):
                overlapping[row, col] = calculate_overlapping_coefficient(
                    blocks_A[row], blocks_A[col]
                )

        i_lower = np.tril_indices(n, k=-1)
        overlapping[i_lower] = overlapping.T[i_lower]
        # A trick learned from https://stackoverflow.com/a/42209263
        return overlapping

MIN_OVERLAPPING_THRESHOLD = 0.65

def remove_overlapping_textual_blocks_for_non_textual_blocks(blocks):
    # Firstly checking paragraph and non-paragraph blocks

    textual_blocks = [b for b in blocks if not is_non_textual_type(b)]
    non_textual_blocks = [b for b in blocks if is_non_textual_type(b)]

    if len(textual_blocks) == 0 or len(non_textual_blocks) == 0:
        return textual_blocks + non_textual_blocks

    overlapping = calculate_pairwise_overlapping_coefficient(
        non_textual_blocks, textual_blocks
    )
    overlapping = overlapping > 0.8

    if not overlapping.any():
        return textual_blocks + non_textual_blocks

    nids, tids = np.where(overlapping)

    return [
        b for idx, b in enumerate(textual_blocks) if idx not in np.unique(tids)
    ] + non_textual_blocks

def find_parent_for_elements(block_layout, token_layout, target_attr="parent"):

    for block in block_layout:
        remaining_tokens = []
        for token in token_layout:
            if is_in(token, block):
                setattr(token, target_attr, int(block.id))
            else:
                remaining_tokens.append(token)

        token_layout = remaining_tokens

    for token in token_layout:
        setattr(token, target_attr, None)

def block_snapping(blocks, tokens):

    for block in blocks:
        if is_non_textual_type(block):
            continue
        tokens_in_this_group = []
        for token in tokens:
            if token.parent == block.id:
                tokens_in_this_group.append(token)
        block.block = union_box(tokens_in_this_group).block

def filter_out_overlapping_block(blocks):

    boxes_to_remove = {b.id: 0 for b in blocks}
    for box2 in blocks:
        for box1 in blocks:

            if box1.id == box2.id:
                continue

            if boxes_to_remove[box1.id] or boxes_to_remove[box2.id]:
                continue

            if (
                calculate_overlapping_coefficient(box1, box2)
                > MIN_OVERLAPPING_THRESHOLD
            ):

                if box1.area >= box2.area:
                    boxes_to_remove[box2.id] = 1
                else:
                    boxes_to_remove[box1.id] = 1

    return [b for b in blocks if not boxes_to_remove[b.id]]

def filter_out_overlapping_block_and_union(blocks):

    overlapping = calculate_pairwise_overlapping_coefficient(blocks)
    n_components, labels = connected_components(
        csgraph=overlapping > MIN_OVERLAPPING_THRESHOLD,
        directed=False,
        return_labels=True,
    )

    new_blocks = []
    prev_len = 0
    for idx, gp in groupby(labels):
        cur_len = len(list(gp))
        cur_blocks = blocks[prev_len : prev_len + cur_len]
        new_blocks.append(
            union_box(sorted(cur_blocks, key=lambda b: b.area, reverse=True)).set(
                id=idx
            )
        )
        prev_len += cur_len

    return new_blocks

def is_close(block1, block2, x_tolerance=15, y_tolerance=16):
    # horizontal difference
    bbox0 = block1.coordinates
    bbox1 = block2.coordinates
    if bbox1[0] - bbox0[2] > x_tolerance or bbox0[0] - bbox1[2] > x_tolerance:
        return False

    # line difference
    _, y1 = block1.block.center
    _, y2 = block2.block.center
    if abs(y1 - y2) > y_tolerance:
        return False
    return True

def group_contents(tokens, x_tolerance=15, y_tolerance=16):

    selected_mask = {b.id: 0 for b in tokens}

    grouped_tokens = []

    cur_tokens = tokens

    while cur_tokens:

        current_group = []

        # start from a random sample to improve robustness
        start_token = random.choice(cur_tokens)

        queue = [start_token]
        selected_mask[start_token.id] = 1

        while queue:
            cur_token = queue[0]
            for candidate_token in cur_tokens:
                if not selected_mask[candidate_token.id] and is_close(
                    cur_token, candidate_token, x_tolerance, y_tolerance
                ):
                    queue.append(candidate_token)
                    selected_mask[candidate_token.id] = 1

            current_group.append(queue.pop(0))

        grouped_tokens.append(current_group)
        cur_tokens = [token for token in cur_tokens if not selected_mask[token.id]]

    return grouped_tokens

def group_ungrouped_elements(
    tokens, attr_name="parent", x_tolerance=15, y_tolerance=16
):
    selected_tokens = [
        b
        for b in tokens
        if getattr(b, attr_name, None) is None
    ]

    results = group_contents(selected_tokens, x_tolerance, y_tolerance)
    return [union_box(ele).set(text=" ".join(i.text for i in ele)) for ele in results]


group_token_to_blocks = group_ungrouped_elements
group_token_to_lines = partial(
    group_ungrouped_elements, attr_name="line_id", y_tolerance=5
)

MIN_OVERLAPPING_THRESHOLD = 0.65

def absorb_additional_blocks_into_existing_blocks(
    blocks, additional_blocks, threshold=MIN_OVERLAPPING_THRESHOLD
):

    if len(blocks) == 0 or len(additional_blocks) == 0:
        return blocks, additional_blocks

    overlapping = calculate_pairwise_overlapping_coefficient(additional_blocks, blocks)
    block_indices, add_block_indices = np.where(overlapping.T >= threshold)
    # Ensure the block_indices are appropriately ordered

    if len(add_block_indices) == 0:
        return blocks, additional_blocks
    else:
        block_ids_to_remove = []
        additional_block_ids_to_remove = []
        newly_added_blocks = []

        prev_len = 0
        for orig_idx, gp in groupby(block_indices):
            cur_len = len(list(gp))
            additional_block_indices_in_this_group = add_block_indices[
                prev_len : prev_len + cur_len
            ]
            block_ids_to_remove.append(orig_idx)
            additional_block_ids_to_remove.extend(
                additional_block_indices_in_this_group
            )

            newly_added_blocks.append(
                union_box(
                    [blocks[orig_idx]]
                    + [
                        additional_blocks[ad_idx]
                        for ad_idx in additional_block_indices_in_this_group
                    ]
                )
                # it will keep the category from the original block
            )
            prev_len += cur_len

        # for ad_idx, orig_idx in zip(add_block_indices, block_indices):
        #     if overlapping[ad_idx, orig_idx] < MIN_OVERLAPPING_THRESHOLD:
        #         continue

        #     block_ids_to_remove.append(orig_idx)
        #     additional_block_ids_to_remove.append(orig_idx)

        #     newly_added_blocks.append(
        #         union_box([blocks[orig_idx], additional_blocks[ad_idx]])
        #         # it will keep the category from the original block
        #     )

        # for ad_idx, orig_idx in zip(add_block_indices, block_indices):
        #     if overlapping[ad_idx, orig_idx] < MIN_OVERLAPPING_THRESHOLD:
        #         continue

        #     block_ids_to_remove.append(orig_idx)
        #     additional_block_ids_to_remove.append(orig_idx)

        #     newly_added_blocks.append(
        #         union_box([blocks[orig_idx], additional_blocks[ad_idx]])
        #         # it will keep the category from the original block
        #     )
        return (
            [b for idx, b in enumerate(blocks) if idx not in block_ids_to_remove],
            [
                b
                for idx, b in enumerate(additional_blocks)
                if idx not in additional_block_ids_to_remove
            ]
            + newly_added_blocks,
        )

def find_parent_for_all_elements_and_reassign_block_category(
    block_layout, token_layout, target_attr="parent", block_ordering_method="token_id"
):

    assert len(block_layout) > 0

    block_min_token_ids = []

    iterating_tokens = token_layout
    for idx, block in enumerate(block_layout):
        block.id = idx
        remaining_tokens = []
        tokens_in_this_block = []
        block_min_token_id = float("inf")
        for token in iterating_tokens:
            if is_in(token, block):
                setattr(token, target_attr, idx)
                tokens_in_this_block.append(token)
                block_min_token_id = min(token.id, block_min_token_id)
            else:
                remaining_tokens.append(token)

        iterating_tokens = remaining_tokens
        # if is_non_textual_type(block):
        #     for token in tokens_in_this_block:
        #         token.type = block.type
        # else:
        #     token_types_in_this_block = [b.type for b in tokens_in_this_block]
        #     block.type = get_most_common_element(token_types_in_this_block)

        block_min_token_ids.append(block_min_token_id)

    if block_ordering_method == "token_id":
        sorted_block_token_indices = {
            orig_id: new_id
            for new_id, orig_id in enumerate(argsort(block_min_token_ids))
        }

    for token in token_layout:
        setattr(
            token,
            target_attr,
            sorted_block_token_indices.get(getattr(token, target_attr, None), None),
        )
    for block in block_layout:
        block.id = sorted_block_token_indices[block.id]

    # print(
    #     f"Searching the closet blocks for the remaining {len(remaining_tokens)} tokens"
    # )
    for token in remaining_tokens:
        setattr(
            token, target_attr, int(find_closet_block_for_token(token, block_layout).id)
        )

def find_minimum_gap(block_A, block_B):
    # just the manhattan distance

    center_A = block_A.block.center
    center_B = block_B.block.center
    return sum(abs(a - b) for a, b in zip(center_A, center_B))

def find_closet_block_for_token(token, blocks):
    gap = float("inf")
    target_block = None
    for block in blocks:
        cur_gap = find_minimum_gap(token, block)
        if cur_gap < gap:
            gap = cur_gap
            target_block = block
    assert target_block is not None
    return target_block

def intersect(self, other):
    return lp.Rectangle(
        max(self.x_1, other.x_1),
        max(self.y_1, other.y_1),
        min(self.x_2, other.x_2),
        min(self.y_2, other.y_2),
    )

def trim_elements_based_on_parents(block_layout, token_layout):

    block_layout = {b.id: b for b in block_layout}

    for token in token_layout:
        block = block_layout.get(token.parent, None)
        if block is not None:
            token.block = intersect(token.block, block.block)

def get_tokens_in_block(tokens, block, metric="center"):

    return [tok for tok in tokens if is_in(tok, block, metric=metric)]

def reorder_lines(lines, blocks, tokens):
    # We firstly group lines by blocks, then order
    # lines within each group using the token indices

    ordered_blocks = sorted(blocks, key=lambda b: b.id)

    tokens_groupby_blocks = {
        block.id: [tok for tok in tokens if tok.parent == block.id] for block in blocks
    }

    for token in tokens:
        token.line_id = None

    line_id = 0
    iter_lines = lines

    for block in ordered_blocks:

        lines_in_current_block = []
        remaining_lines = []
        for line in iter_lines:
            if is_in(line, block):
                lines_in_current_block.append(line)
            else:
                remaining_lines.append(line)
        iter_lines = remaining_lines

        tokens_in_current_block = tokens_groupby_blocks[block.id]

        tokens_in_each_line = [
            get_tokens_in_block(tokens_in_current_block, line, metric="any")
            for line in lines_in_current_block
        ]

        min_token_indices_in_each_line = [
            (min(tok.id for tok in tokens) if len(tokens) > 0 else float("inf"))
            for tokens in tokens_in_each_line
        ]

        # print(min_token_indices_in_each_line)

        sorted_line_token_indices = {
            orig_id: new_id
            for new_id, orig_id in enumerate(argsort(min_token_indices_in_each_line))
        }

        used_line_id = 0
        for idx, line in enumerate(lines_in_current_block):

            tokens_in_this_line = tokens_in_each_line[idx]

            if len(tokens_in_this_line) == 0:
                line.id = None
                continue

            line.id = cur_line_id = sorted_line_token_indices[idx] + line_id
            line.parent = block.id
            line.type = get_most_common_token_type(tokens_in_this_line)
            used_line_id += 1

            for token in tokens_in_this_line:
                token.line_id = cur_line_id

        line_id += used_line_id

    # print(
    #     "Searching the closet blocks for the remaining", len(remaining_lines), "lines"
    # )

    for line in remaining_lines:
        tokens_in_this_line = get_tokens_in_block(tokens, line, metric="coef")
        if len(tokens_in_this_line) == 0:
            line.id = None
        else:
            block = find_closet_block_for_token(line, blocks)
            line.id = cur_line_id = line_id
            line.parent = block.id
            line.type = get_most_common_token_type(tokens_in_this_line)
            for token in tokens_in_this_line:
                token.line_id = cur_line_id

            line_id += 1

    tokens_without_line_ids = [token for token in tokens if token.line_id is None]
    # print(
    #     "Searching the closet lines for the remaining", len(tokens_without_line_ids), "tokens"
    # )
    for token in tokens_without_line_ids:
        line = find_closet_block_for_token(token, lines)
        token.line_id = line.id

def replace_non_text_lines_with_block(lines, blocks, tokens):

    blocks = {b.id: b for b in blocks}

    new_line_id = 0
    synthesized_lines = []
    line_id_conversion = {}

    for bid, gp in groupby(lines, key=lambda ele: ele.parent):
        lines_in_this_block = list(gp)
        cur_block = blocks[bid]
        if is_non_textual_type(cur_block):
            cur_block = copy(cur_block)

            cur_block.parent = cur_block.id
            cur_block.id = new_line_id
            synthesized_lines.append(cur_block)

            for line in lines_in_this_block:
                line_id_conversion[line.id] = new_line_id

            new_line_id += 1

        else:
            for idx, line in enumerate(lines_in_this_block, start=new_line_id):
                line_id_conversion[line.id] = idx
                line.id = idx
            synthesized_lines.extend(lines_in_this_block)
            new_line_id = idx + 1

    for token in tokens:
        if token.line_id not in line_id_conversion:
            for line in synthesized_lines:
                if is_in(token, line, metric="coef"):
                    token.line_id = line.id
                    continue
            if token.line_id is None:
                line = find_closet_block_for_token(token, synthesized_lines)
                token.line_id = line.id
        else:
            token.line_id = line_id_conversion[token.line_id]

    return synthesized_lines


def pipeline(base_path, pdf_sha, pid):
    
    csv_name = f"{pdf_sha}-{pid}.csv"
    tokens = load_cermine_data_from_csv(f'{base_path}/tokens/{csv_name}')

    if tokens is None or len(tokens) == 0:
        # Nothing for empty tokens
        return [[]]*5 

    blocks = load_block_data_from_csv(f'{base_path}/blocks/{csv_name}')
    lines = load_line_data_from_csv(f'{base_path}/lines/{csv_name}')

    blocks = remove_overlapping_textual_blocks_for_non_textual_blocks(blocks)
    find_parent_for_elements(blocks, tokens)
    block_snapping(blocks, tokens)
    blocks = filter_out_overlapping_block_and_union(blocks)

    find_parent_for_elements(blocks, tokens)

    additional_blocks = group_token_to_blocks(tokens)
    blocks, additional_blocks = absorb_additional_blocks_into_existing_blocks(
        blocks, additional_blocks
    )
    
    find_parent_for_all_elements_and_reassign_block_category(
        blocks + additional_blocks, tokens
    )

    lines = filter_out_overlapping_block(lines)
    find_parent_for_elements(blocks, lines)
    trim_elements_based_on_parents(blocks, lines)

    find_parent_for_elements(lines, tokens, target_attr="line_id")
    additional_lines = group_token_to_lines(tokens)
    lines, additional_lines = absorb_additional_blocks_into_existing_blocks(
        lines, additional_lines, threshold=0.3
    )
    reorder_lines(lines + additional_lines, blocks + additional_blocks, tokens)

    blocks = sorted([ele for ele in blocks + additional_blocks], key=lambda ele: ele.id)
    lines = sorted(
        [ele for ele in lines + additional_lines if ele.id is not None],
        key=lambda ele: ele.id,
    )
    lines = replace_non_text_lines_with_block(lines, blocks, tokens)

    return (blocks, lines, tokens, additional_blocks, additional_lines)

def create_structure_df(tokens, blocks, lines):
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
        for ele in blocks
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
        for ele in lines
    ]
    tokens_to_save = [
        [
            ele.id, 
            *ele.coordinates,
            ele.text,
            ele.type,
            ele.parent,
            ele.line_id,
            False,
            False,
        ]
        for ele in tokens
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
    return df