from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class HierarchicalModelConfig(PretrainedConfig):

    model_type = "hierarchical_model"

    def __init__(
        self,
        vocab_size=30522,
        textline_encoder_type="bert-layer",
        textline_model_type="bert-layer",
        textline_encoder_output="cls",
        load_weights_from_existing_model=False,
        textline_encoder_used_bert_layer="first",
        textline_model_used_bert_layer="first",
        pad_token_id=0,
        initializer_range=0.02,
        hidden_dropout_prob=0.1,
        hidden_size=768,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size

        self.textline_encoder_type = textline_encoder_type
        self.textline_model_type = textline_model_type
        self.textline_encoder_output = textline_encoder_output

        self.load_weights_from_existing_model = load_weights_from_existing_model
        self.textline_encoder_used_bert_layer = textline_encoder_used_bert_layer
        self.textline_model_used_bert_layer = textline_model_used_bert_layer

        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range

        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size