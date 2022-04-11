import os
from typing import List, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod
import itertools
import inspect
from dataclasses import dataclass
import dataclasses

import torch
import layoutparser as lp

from .dataset.preprocessors import (
    instantiate_dataset_preprocessor,
    VILAPreprocessorConfig,
)
from .models.hierarchical_model import HierarchicalModelForTokenClassification
from .automodel import AutoModelForTokenClassification, AutoTokenizer

def columns_used_in_model_inputs(model):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    return signature_columns


def flatten_line_level_prediction(batched_line_pred, batched_line_word_count):
    final_flattend_pred = []
    for line_pred, line_word_count in zip(batched_line_pred, batched_line_word_count):
        assert len(line_pred) == len(line_word_count)
        for (pred, label), (line_id, count) in zip(line_pred, line_word_count):
            final_flattend_pred.append([[pred, label, line_id]] * count)

    return list(itertools.chain.from_iterable(final_flattend_pred))


def _clip_bbox(bbox, max_width=1000, max_height=1000):
    """
    A temporary solution to the "large PDF" issue: 
    Instead of normalizing the bounding box, we clip 
    it to a maximum width and height. 

    The bbox format in pdf_data is [x1, y1, x2, y2]. 
    
    #TODO this should be replaced by a normalization 
    #     function that is applied to the bounding box
    """
    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])
    bbox[2] = min(max_width, bbox[2])
    bbox[3] = min(max_height, bbox[3])
    return bbox


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
            preprocessor_config = VILAPreprocessorConfig.from_pretrained(model_path, **preprocessor_config)
            preprocessor = cls.initialize_preprocessor(tokenizer, preprocessor_config)

        return cls(model, preprocessor, device)

    @staticmethod
    @abstractmethod
    def initialize_preprocessor(tokenizer, config):
        pass

    def predict(self, pdf_data) -> lp.Layout:

        model_inputs = self.preprocess_pdf_data(pdf_data)
        model_outputs = self.model(**self.model_input_collator(model_inputs))
        model_predictions = self.get_category_prediction(model_outputs)
        return self.postprocess_model_outputs(pdf_data, model_inputs, model_predictions)

    def get_category_prediction(self, model_outputs):
        predictions = model_outputs.logits.argmax(dim=-1).cpu().detach().numpy()
        return predictions

    def preprocess_pdf_data(self, pdf_data):
        _labels = pdf_data.get("labels")
        pdf_data["labels"] = [0] * len(pdf_data["words"])

        pdf_data["bbox"] = [_clip_bbox(bbox) for bbox in pdf_data["bbox"]] #TODO: Change in the future

        sample = self.preprocessor.preprocess_sample(pdf_data)
        pdf_data["labels"] = _labels
        return sample

    def model_input_collator(self, sample):

        return {
            key: torch.tensor(val, dtype=torch.int64, device=self.device)
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

    def postprocess_model_outputs(self, pdf_data, model_inputs, model_predictions):

        encoded_labels = model_inputs["labels"]

        true_predictions = [
            [(p, l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(model_predictions, encoded_labels)
        ]

        true_predictions = list(itertools.chain.from_iterable(true_predictions))
        preds = [self.id2label.get(ele[0], ele[0]) for ele in true_predictions]
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

    def postprocess_model_outputs(self, pdf_data, model_inputs, model_predictions):

        encoded_labels = model_inputs["labels"]

        true_predictions = [
            [(p, l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(model_predictions, encoded_labels)
        ]

        flatten_predictions = flatten_line_level_prediction(
            true_predictions, model_inputs["group_word_count"]
        )

        preds = [self.id2label.get(ele[0], ele[0]) for ele in flatten_predictions]
        words = pdf_data["words"]
        bboxes = pdf_data["bbox"]

        generated_tokens = []
        for word, pred, bbox in zip(words, preds, bboxes):
            generated_tokens.append(
                lp.TextBlock(block=lp.Rectangle(*bbox), text=word, type=pred)
            )

        return lp.Layout(generated_tokens)
