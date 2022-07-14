from typing import List, Optional, Union, Dict, Any, Tuple
from abc import abstractmethod
import itertools
import inspect
import logging
import copy

import numpy as np
import torch
import layoutparser as lp

from .dataset.preprocessors import (
    instantiate_dataset_preprocessor,
    VILAPreprocessorConfig,
)
from .pdftools.pdfplumber_extractor import PDFPlumberPageData
from .automodel import AutoModelForTokenClassification, AutoTokenizer
from .constants import MODEL_PDF_WIDTH, MODEL_PDF_HEIGHT, UNICODE_CATEGORIES_TO_REPLACE
from .utils import replace_unicode_tokens

logger = logging.getLogger(__name__)

AGG_LEVEL_TO_GROUP_NAME = {
    "row": "line",
    "block": "block",
}


def columns_used_in_model_inputs(model):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    return signature_columns


def flatten_group_level_prediction(batched_group_pred, batched_group_word_count):
    final_flatten_pred = []
    for group_pred, group_word_count in zip(
        batched_group_pred, batched_group_word_count
    ):
        assert len(group_pred) == len(group_word_count)
        for (pred, label), (line_id, count) in zip(group_pred, group_word_count):
            final_flatten_pred.append([[pred, label, line_id]] * count)

    return list(itertools.chain.from_iterable(final_flatten_pred))


def normalize_bbox(
    bbox,
    page_width,
    page_height,
    target_width=MODEL_PDF_WIDTH,
    target_height=MODEL_PDF_HEIGHT,
):
    """
    Normalize bounding box to the target size.
    """

    x1, y1, x2, y2 = bbox

    # Right now only execute this for only "large" PDFs
    # TODO: Change it for all PDFs


    if x1 > x2:
        logger.debug(f"Incompatible x coordinates: x1:{x1} > x2:{x2}")
        x1, x2 = x2, x1

    if y1 > y2:
        logger.debug(f"Incompatible y coordinates: y1:{y1} > y2:{y2}")
        y1, y2 = y2, y1

    if page_width > target_width or page_height > target_height:

        # Aspect ratio preserving scaling
        scale_factor = target_width / page_width if page_width > page_height else target_height / page_height

        logger.debug(f"Scaling page as page width {page_width} is larger than target width {target_width} or height {page_height} is larger than target height {target_height}")
        
        x1 = float(x1) * scale_factor
        x2 = float(x2) * scale_factor

        y1 = float(y1) * scale_factor
        y2 = float(y2) * scale_factor

    return (x1, y1, x2, y2)


def unnormalize_bbox(
    bbox,
    page_width,
    page_height,
    target_width=MODEL_PDF_WIDTH,
    target_height=MODEL_PDF_HEIGHT,
):
    """
    Unnormalize bounding box to the target size.
    """

    x1, y1, x2, y2 = bbox

    # Right now only execute this for only "large" PDFs
    # TODO: Change it for all PDFs
    
    if page_width > target_width or page_height > target_height:

        # Aspect ratio preserving scaling
        scale_factor = target_width / page_width if page_width > page_height else target_height / page_height

        logger.debug(f"Scaling page as page width {page_width} is larger than target width {target_width} or height {page_height} is larger than target height {target_height}")
        
        x1 = float(x1) / scale_factor
        x2 = float(x2) / scale_factor

        y1 = float(y1) / scale_factor
        y2 = float(y2) / scale_factor

    return (x1, y1, x2, y2)


class BasePDFPredictor:
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.id2label = self.model.config.id2label

        if device is None:
            self.device = model.device
        else:
            self.device = device
            model.to(self.device)

        self.model.eval()
        self._used_cols = columns_used_in_model_inputs(self.model)

    @classmethod
    def from_pretrained(
        cls, model_path, preprocessor=None, device=None, **preprocessor_config
    ):

        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if preprocessor is None:
            preprocessor_config = VILAPreprocessorConfig.from_pretrained(
                model_path, **preprocessor_config
            )
            preprocessor = cls.initialize_preprocessor(tokenizer, preprocessor_config)

        return cls(model, preprocessor, device)

    @staticmethod
    @abstractmethod
    def initialize_preprocessor(tokenizer, config):
        pass

    def predict(
        self,
        page_data: Dict,
        page_size: Tuple,
        batch_size: Optional[int] = None,
        return_type: Optional[str] = "layout",
        replace_empty_unicode: Optional[bool] = True,
    ) -> Union[lp.Layout, List]:
        """This is a generalized predict function that runs vila on a PDF page.

        Args:
            page_data (Dict):
                The page-level data as a dict in the form of
                {
                    'words': ['word1', 'word2', ...],
                    'bbox': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                    'block_ids': [0, 0, 0, 1 ...],
                    'line_ids': [0, 1, 1, 2 ...],
                    'labels': [0, 0, 0, 1 ...], # could be empty
                }
            page_size (Tuple):
                A tuple of (width, height) for this page
            batch_size (Optional[int]):
                Specifying the maximum number of batches for each model run.
                By default it will encode all pages all at once.
            return_type (Optional[str]):
                It can be either "layout", for a structured token output,
                or "list" for a list of predicted classes. Default is "layout".
            replace_empty_unicode (Optional[bool]):
                If True, replace certain unicode tokens with the "unknown" token
                from the tokenizer. Default is True.
        """

        # page_size is (page_token.width, page_token.height)
        model_inputs = self.preprocess_pdf_data(
            page_data, page_size, replace_empty_unicode
        )
        batched_inputs = self.model_input_collator(model_inputs, batch_size)

        model_predictions = []
        for batch in batched_inputs:
            model_outputs = self.model(**batch)
            model_predictions.append(self.get_category_prediction(model_outputs))

        model_predictions = np.vstack(model_predictions)
        return self.postprocess_model_outputs(
            page_data, model_inputs, model_predictions, return_type
        )

    def predict_page(
        self,
        page_tokens: PDFPlumberPageData,
        page_image: Optional["PIL.Image"] = None,
        visual_group_detector: Optional[Any] = None,
        page_size=None,
        batch_size=None,
        return_type: Optional[str] = "layout",
        replace_empty_unicode: Optional[bool] = True,
    ) -> Union[lp.Layout, List]:
        """The predict_page function is used for running the model on a single page
        in the vila page_token objects.

        Args:
            page_tokens (PDFPlumberPageData):
                The page-level data as an PDFPlumberPageData object.
            visual_group_detector:
                The visual group model to use for detecting the required visual groups.
            page_size (Tuple):
                A tuple of (width, height) for this page. By default it will use the
                page_size from the page_tokens directly unless the page_size is explicitly
                specified.
            batch_size (Optional[int]):
                Specifying the maximum number of batches for each model run.
                By default it will encode all pages all at once.
            replace_empty_unicode (Optional[bool]):
                If True, replace certain unicode tokens with the "unknown" token
                from the tokenizer. Default is True.
        """
        page_tokens = copy.copy(page_tokens)
        required_agg_level = self.preprocessor.config.agg_level
        required_group = AGG_LEVEL_TO_GROUP_NAME[required_agg_level]

        if not getattr(page_tokens, required_group + "s"):  # either none or empty
            if page_image is not None and visual_group_detector is not None:
                logger.warning(
                    f"The required_group {required_group} is missing in page_tokens."
                    f"Using the page_image and visual_group_detector to detect."
                )
                detected_groups = visual_group_detector.detect(page_image)
                page_tokens.annotate(**{required_group + "s": detected_groups})
            else:
                raise ValueError(
                    f"The required_group {required_group} is missing in page_tokens."
                )

        pdf_data = page_tokens.to_pagedata().to_dict()
        predicted_tokens = self.predict(
            page_data=pdf_data,
            page_size=page_tokens.page_size if page_size is None else page_size,
            batch_size=batch_size,
            return_type=return_type,
            replace_empty_unicode=replace_empty_unicode,
        )

        return predicted_tokens

    def get_category_prediction(self, model_outputs):
        predictions = model_outputs.logits.argmax(dim=-1).cpu().detach().numpy()
        return predictions

    def preprocess_pdf_data(self, pdf_data, page_size, replace_empty_unicode):
        _labels = pdf_data.get("labels")
        pdf_data["labels"] = [0] * len(pdf_data["words"])
        page_width, page_height = page_size

        _words = pdf_data["words"]
        if replace_empty_unicode:
            pdf_data["words"] = replace_unicode_tokens(
                pdf_data["words"],
                UNICODE_CATEGORIES_TO_REPLACE,
                self.preprocessor.tokenizer.unk_token,
            )

        _bbox = pdf_data["bbox"]
        pdf_data["bbox"] = [
            normalize_bbox(box, page_width, page_height) for box in pdf_data["bbox"]
        ]
        sample = self.preprocessor.preprocess_sample(pdf_data)

        # Change back to the original pdf_data
        pdf_data["labels"] = _labels
        pdf_data["words"] = _words
        pdf_data["bbox"] = _bbox

        return sample

    def model_input_collator(self, sample, batch_size=None):

        n_samples = len(sample[list(sample.keys())[0]])
        if batch_size is None:
            batch_size = len(sample[list(sample.keys())[0]])

        for idx in range(0, n_samples, batch_size):
            yield {
                key: torch.tensor(val[idx : idx + batch_size], device=self.device).type(
                    torch.int64
                )  # Make the conversion more robust
                for key, val in sample.items()
                if key in self._used_cols
            }

    @abstractmethod
    def postprocess_model_outputs(self, pdf_data, model_inputs, model_predictions):
        pass


class SimplePDFPredictor(BasePDFPredictor):
    """The PDF predictor used for basic models like BERT or LayoutLM."""

    @staticmethod
    def initialize_preprocessor(tokenizer, config):
        return instantiate_dataset_preprocessor("base", tokenizer, config)

    def postprocess_model_outputs(
        self, pdf_data, model_inputs, model_predictions, return_type
    ):

        encoded_labels = model_inputs["labels"]

        true_predictions = [
            [(p, l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(model_predictions, encoded_labels)
        ]

        true_predictions = list(itertools.chain.from_iterable(true_predictions))
        preds = [self.id2label.get(ele[0], ele[0]) for ele in true_predictions]

        assert len(preds) == len(pdf_data["words"])
        if return_type == "list":
            return preds

        elif return_type == "layout":
            words = [pdf_data["words"][idx] for idx in model_inputs["encoded_word_ids"]]
            bboxes = [pdf_data["bbox"][idx] for idx in model_inputs["encoded_word_ids"]]

            generated_tokens = []
            for word, pred, bbox in zip(words, preds, bboxes):
                generated_tokens.append(
                    lp.TextBlock(block=lp.Rectangle(*bbox), text=word, type=pred)
                )

            return lp.Layout(generated_tokens)


class LayoutIndicatorPDFPredictor(SimplePDFPredictor):
    """The PDF predictor used for layout indicator, or IVILA, based models.
    Right now, the postprocess_model_outputs is identical to that in SimplePDFPredictor"""

    @staticmethod
    def initialize_preprocessor(tokenizer, config):
        return instantiate_dataset_preprocessor("layout_indicator", tokenizer, config)


class HierarchicalPDFPredictor(BasePDFPredictor):
    """The PDF predictor used for hierarchical, or HVILA, based models."""

    @staticmethod
    def initialize_preprocessor(tokenizer, config):
        return instantiate_dataset_preprocessor(
            "hierarchical_modeling", tokenizer, config
        )

    def postprocess_model_outputs(
        self, pdf_data, model_inputs, model_predictions, return_type
    ):

        encoded_labels = model_inputs["labels"]

        true_predictions = [
            [(p, l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(model_predictions, encoded_labels)
        ]

        flatten_predictions = flatten_group_level_prediction(
            true_predictions, model_inputs["group_word_count"]
        )

        preds = [self.id2label.get(ele[0], ele[0]) for ele in flatten_predictions]
        # We don't need assertion here because flatten_group_level_prediction
        # already guarantees the length of preds is the same as pdf_data["words"]

        if return_type == "list":
            return preds

        elif return_type == "layout":
            words = pdf_data["words"]
            bboxes = pdf_data["bbox"]

            generated_tokens = []
            for word, pred, bbox in zip(words, preds, bboxes):
                generated_tokens.append(
                    lp.TextBlock(block=lp.Rectangle(*bbox), text=word, type=pred)
                )

            return lp.Layout(generated_tokens)
