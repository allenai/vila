from typing import List, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod
import itertools
import inspect
from dataclasses import dataclass
import dataclasses

import torch
import layoutparser as lp
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..dataset.preprocessors import instantiate_dataset_preprocessor


def columns_used_in_model_inputs(model):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    return signature_columns


@dataclass
class PreprocessorConfig:
    agg_level: str = "row"
    label_all_tokens: bool = False
    group_bbox_agg: str = "first"
    added_special_sepration_token: str = "[SEP]"


class BasePredictor:
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor

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
            preprocessor = cls.initialize_preprocessor(
                tokenizer, PreprocessorConfig(**preprocessor_config)
            )

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
        predictions = model_outputs.logits.argmax(dim=-1).cpu().numpy()
        return predictions

    def preprocess_pdf_data(self, pdf_data):
        _labels = pdf_data.get('labels')
        pdf_data['labels']  = [0]*len(pdf_data['words'])

        sample = self.preprocessor.preprocess_sample(pdf_data)
        pdf_data['labels'] = _labels
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


class SimplePDFPredictor(BasePredictor):
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
        preds = [ele[0] for ele in true_predictions]
        words = [pdf_data["words"][idx] for idx in model_inputs["encoded_word_ids"]]
        bboxes = [pdf_data["bbox"][idx] for idx in model_inputs["encoded_word_ids"]]

        generated_tokens = []
        for word, pred, bbox in zip(words, preds, bboxes):
            generated_tokens.append(
                lp.TextBlock(block=lp.Rectangle(*bbox), text=word, type=pred)
            )

        return lp.Layout(generated_tokens)