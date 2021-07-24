from typing import List, Union, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

import layoutparser as lp
from numpy.core.shape_base import block
import pandas as pd

from ..utils import union_lp_box

@dataclass
class PageData:
    blocks: List[lp.TextBlock]
    lines: List[lp.TextBlock]
    words: List[lp.TextBlock]

    def __post_init__(self):
        for w in self.words:
            if not hasattr(w, "block_id"): # type: ignore
                w.block_id = None
            if not hasattr(w, "line_id"): 
                w.line_id = None

        for l in self.words:
            if not hasattr(l, "block_id"):
                l.block_id = None

    def to_dataframe(
        self,
        keep_token_index=True,
        normalize_coordinates=False,
        canvas_width=None,
        canvas_height=None,
    ) -> pd.DataFrame:

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
                ele.block_id,
                -1,
                False,
                True,
            ]
            for ele in self.lines
        ]
        
        tokens_to_save = [
            [
                ele.id if keep_token_index else idx,
                *ele.coordinates,
                ele.text,
                ele.type,
                ele.block_id,
                ele.line_id,
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

    def to_dict(
        self,
        keep_token_index=True,
        normalize_coordinates=False,
        canvas_width=None,
        canvas_height=None,
        category_map: Dict = None,
    ) -> Dict:

        df = self.to_dataframe(
            keep_token_index=keep_token_index,
            normalize_coordinates=normalize_coordinates,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

        # Only select text bocks
        df = df[~df.is_block & ~df.is_line]

        # Filter empty text
        df = df.dropna(axis=0, subset=["text"])
        df = df[~df.text.str.isspace()]

        if len(df) == 0:
            return None

        df["block_id"] = df["block_id"].fillna(-1).astype("int")
        df["line_id"] = df["line_id"].fillna(-1).astype("int")

        row_item = {
            "words": df["text"].tolist(),
            "bbox": df.apply(
                lambda row: (row["x_1"], row["y_1"], row["x_2"], row["y_2"]), axis=1
            ).tolist(),
            "block_ids": df["block_id"].tolist(),
            "line_ids": df["line_id"].tolist(),
        }

        if category_map is not None:
            row_item["labels"] = df["category"].map(category_map).tolist()
        else:
            row_item["labels"] = df["category"].tolist()

        return row_item

    @classmethod
    def from_dict(cls, json_data, default_line_id=-1, default_block_id=-1):
        
        words = []
        lines = defaultdict(list)
        blocks = defaultdict(list)

        for idx, (word, bbox, line_id, block_id, label) in enumerate(zip(
            json_data["words"], json_data["bbox"], json_data["line_ids"], json_data["block_ids"], json_data["labels"]
        )):
            word = lp.TextBlock(
                id=idx,
                block=lp.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3]),
                text=word,
                type=label,
            )
            word.line_id=line_id
            word.block_id=block_id

            lines[line_id].append(word)
            blocks[block_id].append(word)

            words.append(word)

        lines.pop(default_line_id, None)
        blocks.pop(default_block_id, None)

        lines = [
            union_lp_box(contained_words).set(id=id)
            for id, contained_words in sorted(lines.items())
        ]

        blocks = [
            union_lp_box(contained_words).set(id=id)
            for id, contained_words in sorted(blocks.items())
        ]

        return cls(blocks=blocks, lines=lines, words=words)