import inspect

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    BertPreTrainedModel,
    BertModel,
    LayoutLMModel,
    BertConfig,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss

from ..constants import *
from .configuration_hierarchical_model import HierarchicalModelConfig

MAX_2D_POSITION_EMBEDDINGS = 1024

#TODO: Rename textline to group 
#TODO: Rename encoder and model to group encoder and page encoder 

def instantiate_textline_encoder(config):

    if not config.load_weights_from_existing_model:

        if config.textline_encoder_type == "bert-base-uncased":
            config = AutoConfig.from_pretrained("bert-base-uncased")
            model = AutoModel.from_config(config)
            return model
        elif config.textline_encoder_type == "bert-layer":
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.num_hidden_layers = 1
            model = AutoModel.from_config(config)
            return model
        else:
            raise (
                f"Invalid textline_encoder_type: {config.textline_encoder_type}."
                "Must be one of {bert-base-uncased, bert-layer}"
            )

    else:

        if config.textline_encoder_type in ["bert-layer", "bert-base-uncased"]:
            model = AutoModel.from_pretrained("bert-base-uncased")
            if config.textline_encoder_type == "bert-layer":
                if config.textline_encoder_used_bert_layer == "first":
                    model.encoder.layer = nn.ModuleList([model.encoder.layer[0]])
                elif config.textline_encoder_used_bert_layer == "last":
                    model.encoder.layer = nn.ModuleList([model.encoder.layer[-1]])
            return model
        else:
            raise (
                f"Invalid textline_encoder_type: {config.textline_encoder_type}."
                "Must be one of {bert-base-uncased, bert-layer}"
            )


def instantiate_textline_model(config):
    if not config.load_weights_from_existing_model:
        if config.textline_model_type == "bert-base-uncased":
            config = AutoConfig.from_pretrained("bert-base-uncased")
            model = AutoModel.from_config(config)
            return model
        elif config.textline_model_type == "bert-layer":
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.num_hidden_layers = 1
            model = AutoModel.from_config(config)
            return model
        elif "layoutlm-base-uncased" in config.textline_model_type:
            config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
            model = AutoModel.from_config(config)
            return model
        elif config.textline_model_type == "layoutlm-layer":
            config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
            config.num_hidden_layers = 1
            model = AutoModel.from_config(config)
            return model
        else:
            raise (
                f"Invalid textline_model_type: {config.textline_model_type}."
                "Must be one of {bert-base-uncased, bert-layer, "
                "layoutlm-base-uncased, layoutlm-layer}"
            )
    else:
        if config.textline_model_type in ["bert-layer", "bert-base-uncased"]:
            model = AutoModel.from_pretrained("bert-base-uncased")
            if config.textline_model_type == "bert-layer":
                if config.textline_model_used_bert_layer == "first":
                    model.encoder.layer = nn.ModuleList([model.encoder.layer[0]])
                elif config.textline_model_used_bert_layer == "last":
                    model.encoder.layer = nn.ModuleList([model.encoder.layer[-1]])
            return model
        elif config.textline_model_type in [
            "layoutlm-layer",
            "layoutlm-base-uncased",
            "microsoft/layoutlm-base-uncased",
        ]:
            model = AutoModel.from_pretrained("microsoft/layoutlm-base-uncased")
            if config.textline_model_type == "layoutlm-layer":
                if config.textline_model_used_bert_layer == "first":
                    model.encoder.layer = nn.ModuleList([model.encoder.layer[0]])
                elif config.textline_model_used_bert_layer == "last":
                    model.encoder.layer = nn.ModuleList([model.encoder.layer[-1]])
            return model
        else:
            raise (
                f"Invalid textline_encoder_type: {config.textline_encoder_type}."
                "Must be one of {bert-base-uncased, bert-layer}"
            )


class HierarchicalPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HierarchicalModelConfig
    base_model_prefix = "hierarchical_model"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SimpleHierarchicalModel(HierarchicalPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.textline_encoder = instantiate_textline_encoder(config)
        self.textline_model = instantiate_textline_model(config)

        self.textline_encoder_output = config.textline_encoder_output
        self.use_bbox_for_textline_model:bool = self._check_use_bbox_for_textline_model()
        if not config.load_weights_from_existing_model:
            self.init_weights()

    def _check_use_bbox_for_textline_model(self) -> bool:
        # TODO: Explain
        signature = inspect.signature(self.textline_model.forward)
        signature_columns = list(signature.parameters.keys())
        return "bbox" in signature_columns
    
    # TODO: Add bbox to textline encoder

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        group_level_attention_mask=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Preprocess the input
        #TODO: Change Ls to something about groups
        B, Ls, Ts = input_ids.shape  # Lines, Tokens
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None: #TODO: Change this name to group_level_attention_mask
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if group_level_attention_mask is None: # TODO: page_level_attention_mask
            attention_mask = torch.ones((B, Ls), device=device)
        if bbox is None:
            bbox = torch.zeros(
                tuple(list(input_shape)[:-1] + [4]), dtype=torch.long, device=device
            )
            # There are 4 elements for each bounding box (left, top, right, bottom)
            # Currently we only accept line-level bbox

        input_ids = input_ids.reshape(B * Ls, -1)
        attention_mask = attention_mask.reshape(B * Ls, -1)
        token_type_ids = token_type_ids.reshape(B * Ls, -1)
        encoded_textlines = self.textline_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # (B * Ls, Ts, hidden_state)

        # TODO: Change this to a pooling method 
        if self.textline_encoder_output.lower() == "cls":
            encoded_textlines = encoded_textlines.last_hidden_state[:, 0, :]
        elif self.textline_encoder_output.lower() == "sep":
            # the sep token is 102 by default
            # for debugging 
            # _, ys = torch.where(input_ids == 102)
            # print(ys) 
            encoded_textlines = encoded_textlines.last_hidden_state[input_ids == 102]
            #TODO: Move this into config_class
        elif self.textline_encoder_output.lower() == "last":
            encoded_textlines = encoded_textlines.last_hidden_state[:, -1, :]
        elif self.textline_encoder_output.lower() == "average":
            encoded_textlines = encoded_textlines.last_hidden_state.mean(dim=1)

        encoded_textlines = encoded_textlines.reshape(B, Ls, -1)

        if self.use_bbox_for_textline_model:
            embedded_lines = self.textline_model.embeddings(
                inputs_embeds=encoded_textlines, bbox=bbox
            )
        else:
            embedded_lines = self.textline_model.embeddings(
                inputs_embeds=encoded_textlines
            )

        outputs = self.textline_model(
            inputs_embeds=embedded_lines, attention_mask=group_level_attention_mask
        )
        return outputs


class HierarchicalModelForTokenClassification(HierarchicalPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.num_labels = config.num_labels
        self.hierarchical_model = SimpleHierarchicalModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,  # This shouldn't be none
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        group_level_attention_mask=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.hierarchical_model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            group_level_attention_mask=group_level_attention_mask,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if group_level_attention_mask is not None:
                active_loss = group_level_attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )