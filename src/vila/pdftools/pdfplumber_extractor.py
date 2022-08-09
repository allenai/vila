"""A Minimal PDFPlumber Parser
MODIFIED FROM https://github.com/allenai/pacer-docket-parser/blob/master/src/docketparser/datamodel.py
"""
from typing import List, Union, Dict, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd
import pdfplumber
import layoutparser as lp

from ..utils import union_lp_box, assign_tokens_to_blocks
from .base import BasePDFTokenExtractor
from .datamodel import PageData


@dataclass
class PDFPlumberPageData:
    height: Union[float, int]
    width: Union[float, int]
    tokens: lp.Layout
    url_tokens: lp.Layout = field(default_factory=list)
    lines: lp.Layout = field(default_factory=list)
    blocks: lp.Layout = field(default_factory=list)

    def get_text_segments(self, x_tolerance=10, y_tolerance=10) -> List[List]:
        """Get text segments from the current page.
        It will automatically add new lines for 1) line breaks
        2) big horizontal gaps
        """
        prev_y = None
        prev_x = None

        lines = []
        token_in_this_line = []
        n = 0

        for token in self.tokens:
            cur_y = token.block.center[1]
            cur_x = token.coordinates[0]

            if prev_y is None:
                prev_y = cur_y
                prev_x = cur_x

            if abs(cur_y - prev_y) <= y_tolerance and cur_x - prev_x <= x_tolerance:

                token_in_this_line.append(token)
                if n == 0:
                    prev_y = cur_y
                else:
                    prev_y = (prev_y * n + cur_y) / (n + 1)
                n += 1

            else:
                lines.append(token_in_this_line)
                token_in_this_line = [token]
                n = 1
                prev_y = cur_y

            prev_x = token.coordinates[2]

        if token_in_this_line:
            lines.append(token_in_this_line)

        return lines

    def get_text(self, x_tolerance=10, y_tolerance=10) -> str:
        """Returns the page text by instering '\n' between text segments
        returned by `self.get_text_segments` .
        """

        return "\n".join(
            [
                " ".join([e.text for e in ele])
                for ele in self.get_text_segments(x_tolerance, y_tolerance)
            ]
        )

    def get_lines(self, x_tolerance=10, y_tolerance=10) -> lp.Layout:
        """Get the text line bounding boxes from the current page."""

        lines = []
        for idx, line_tokens in enumerate(
            self.get_text_segments(x_tolerance, y_tolerance)
        ):
            line = union_lp_box(line_tokens).set(id=idx)
            lines.append(line)
            for t in line_tokens:
                t.line_id = idx

                if not hasattr(t, "block_id"):
                    t.block_id = None
                line.block_id = t.block_id

        return lp.Layout(lines)

    def annotate(self, **kwargs):

        for key, blocks in kwargs.items():
            if key in ["lines", "blocks"]:
                blocks, tokens = assign_tokens_to_blocks(blocks, self.tokens, keep_empty_blocks=True)
                setattr(self, key, blocks)
                self.tokens = tokens

    def to_pagedata(self, x_tolerance=10, y_tolerance=10):
        """Convert the layout to a PageData object."""

        if len(self.lines) == 0:
            lines = self.get_lines(x_tolerance, y_tolerance)
        else:
            lines = self.lines

        return PageData(words=self.tokens, lines=lines, blocks=self.blocks)

    @property
    def page_size(self):
        return (self.width, self.height)


def convert_token_dict_to_layout(tokens):

    lp_tokens = []
    for token in tokens:
        lp_token = lp.TextBlock(
            lp.Rectangle(
                x_1=token["x"],
                y_1=token["y"],
                x_2=token["x"] + token["width"],
                y_2=token["y"] + token["height"],
            ),
            text=token["text"],
        )
        lp_token.font = token.get("font")
        lp_tokens.append(lp_token)

    return lp.Layout(lp_tokens)


def load_page_data_from_dict(source_data: Dict[str, Any]) -> List[Dict]:

    page_data = [
        PDFPlumberPageData(
            height=page_data["page"]["height"],
            width=page_data["page"]["width"],
            tokens=convert_token_dict_to_layout(page_data["tokens"]),
            url_tokens=convert_token_dict_to_layout(page_data["url_tokens"]),
            lines=lp.Layout(
                [
                    lp.Rectangle(
                        x_1=line["x"],
                        y_1=line["y"],
                        x_2=line["x"] + line["width"],
                        y_2=line["y"] + line["height"],
                    )
                    for line in page_data["lines"]
                ]
            ),
        )
        for page_data in source_data
    ]

    for page_token in page_data:
        page_token.lines = page_token.get_lines()

    return page_data


class PDFPlumberTokenExtractor(BasePDFTokenExtractor):
    NAME = "pdfplumber"

    UNDERLINE_HEIGHT_THRESHOLD = 3
    UNDERLINE_WIDTH_THRESHOLD = 100
    # Defines what a regular underline should look like

    @staticmethod
    def convert_to_pagetoken(row: pd.Series) -> Dict:
        """Convert a row in a DataFrame to pagetoken"""
        return dict(
            text=row["text"],
            x=row["x0"],
            width=row["width"],
            y=row["top"],
            height=row["height"],
            font=f'{row.get("fontname")}-{int(row.get("size")) if row.get("size") else ""}',
        )

    def obtain_word_tokens(self, cur_page: pdfplumber.page.Page) -> List[Dict]:
        """Obtain all words from the current page.
        Args:
            cur_page (pdfplumber.page.Page):
                the pdfplumber.page.Page object with PDF token information
        Returns:
            List[PageToken]:
                A list of page tokens stored in PageToken format.
        """
        words = cur_page.extract_words(
            x_tolerance=1.5,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            horizontal_ltr=True,
            vertical_ttb=True,
            extra_attrs=["fontname", "size"],
        )

        if len(words) == 0:
            return []

        df = pd.DataFrame(words)

        # Avoid boxes outside the page
        df[["x0", "x1"]] = (
            df[["x0", "x1"]].clip(lower=0, upper=int(cur_page.width)).astype("float")
        )
        df[["top", "bottom"]] = (
            df[["top", "bottom"]]
            .clip(lower=0, upper=int(cur_page.height))
            .astype("float")
        )

        df["height"] = df["bottom"] - df["top"]
        df["width"] = df["x1"] - df["x0"]

        word_tokens = df.apply(self.convert_to_pagetoken, axis=1).tolist()
        return word_tokens

    def obtain_page_hyperlinks(self, cur_page: pdfplumber.page.Page) -> List[Dict]:

        if len(cur_page.hyperlinks) == 0:
            return []

        df = pd.DataFrame(cur_page.hyperlinks)
        df[["x0", "x1"]] = (
            df[["x0", "x1"]].clip(lower=0, upper=int(cur_page.width)).astype("float")
        )
        df[["top", "bottom"]] = (
            df[["top", "bottom"]]
            .clip(lower=0, upper=int(cur_page.height))
            .astype("float")
        )
        df[["height", "width"]] = df[["height", "width"]].astype("float")

        hyperlink_tokens = (
            df.rename(columns={"uri": "text"})
            .apply(self.convert_to_pagetoken, axis=1)
            .tolist()
        )
        return hyperlink_tokens

    def obtain_page_lines(self, cur_page: pdfplumber.page.Page) -> List[Dict]:

        height = float(cur_page.height)
        page_objs = cur_page.rects + cur_page.lines
        possible_underlines = [
            dict(
                x=float(ele["x0"]),
                y=height - float(ele["y0"]),
                height=float(ele["height"]),
                width=float(ele["width"]),
            )
            for ele in filter(
                lambda obj: obj["height"] < self.UNDERLINE_HEIGHT_THRESHOLD
                and obj["width"] < self.UNDERLINE_WIDTH_THRESHOLD,
                page_objs,
            )
        ]
        return possible_underlines

    def extract(self, pdf_path: str) -> List[Dict]:
        """Extracts token text, positions, and style information from a PDF file.
        Args:
            pdf_path (str): the path to the pdf file.
            include_lines (bool, optional): Whether to include line tokens. Defaults to False.
            target_data (str, optional): {"token", "hyperlink"}
        Returns:
            PdfAnnotations: A `PdfAnnotations` containing all the paper token information.
        """
        plumber_pdf_object = pdfplumber.open(pdf_path)

        pages = []
        for page_id in range(len(plumber_pdf_object.pages)):
            cur_page = plumber_pdf_object.pages[page_id]

            tokens = self.obtain_word_tokens(cur_page)
            url_tokens = self.obtain_page_hyperlinks(cur_page)

            page = dict(
                page=dict(
                    width=float(cur_page.width),
                    height=float(cur_page.height),
                    index=page_id,
                ),
                tokens=tokens,
                url_tokens=url_tokens,
                lines=[],
            )
            pages.append(page)

        return load_page_data_from_dict(pages)
