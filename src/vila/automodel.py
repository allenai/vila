from .models import (
    HierarchicalModelConfig,
    SimpleHierarchicalModel,
    HierarchicalModelForTokenClassification,
)
import warnings

try:
    from transformers import (
        CONFIG_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_NAMES_MAPPING,
        TOKENIZER_MAPPING,
    )

    from transformers.models.auto.modeling_auto import auto_class_factory
    from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer

    warnings.warn("Currently using an old version of transformers for configuring the AutoModel. Consider updating your transformers to the latest version(>=4.21.x).")
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

    AutoModelForTokenClassification = auto_class_factory(
        "AutoModelForTokenClassification",
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        head_doc="token classification",
    )

except:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        AutoModelForTokenClassification,
        BertTokenizer,
        BertTokenizerFast,
    )
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    AutoConfig.register("hierarchical_model", HierarchicalModelConfig)
    CONFIG_MAPPING_NAMES["hierarchical_model"] = HierarchicalModelConfig.__name__
    # An issue in https://github.com/huggingface/transformers/pull/18491
    AutoTokenizer.register(HierarchicalModelConfig, BertTokenizer, BertTokenizerFast)
    AutoModel.register(HierarchicalModelConfig, SimpleHierarchicalModel)
    AutoModelForTokenClassification.register(
        HierarchicalModelConfig, HierarchicalModelForTokenClassification
    )
