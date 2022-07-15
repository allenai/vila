import types

from .models import HierarchicalModelConfig, HierarchicalModelForTokenClassification

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_NAMES_MAPPING,
    TOKENIZER_MAPPING,
)
#from transformers.models.auto.modeling_auto import auto_class_factory
from transformers.models.auto.modeling_auto import _BaseAutoModelClass, auto_class_update
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer

CONFIG_MAPPING.update([("hierarchical_model", HierarchicalModelConfig)])
MODEL_NAMES_MAPPING.update([("hierarchical_model", "HierarchicalModel")])
TOKENIZER_MAPPING.update(
    [
        (HierarchicalModelConfig, (BertTokenizer, BertTokenizerFast)),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(HierarchicalModelConfig, HierarchicalModelForTokenClassification)]
)

cls = types.new_class("AutoModelForTokenClassification", (_BaseAutoModelClass,))
cls._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
cls.__name__ = "AutoModelForTokenClassification"
AutoModelForTokenClassification = auto_class_update(cls, head_doc="token classification")
#AutoModelForTokenClassification = auto_class_factory(
#    "AutoModelForTokenClassification",
#    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
#    head_doc="token classification",
#)
