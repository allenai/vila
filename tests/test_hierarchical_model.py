import sys
import tempfile

import torch
from transformers import AutoModel

from vila.models.configuration_hierarchical_model import HierarchicalModelConfig
from vila.models.hierarchical_model import (
    SimpleHierarchicalModel,
    HierarchicalModelForTokenClassification,
)


def compare_models(model_1, model_2):
    # From https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    return models_differ


def create_bbox(bz, sequence_length):
    _bbox = torch.randint(0, 100, size=(bz, sequence_length, 2))
    bbox_ = torch.randint(100, 200, size=(bz, sequence_length, 2))
    bbox = torch.cat([_bbox, bbox_], dim=-1)
    return bbox


def test_reload_hierarchical_model():

    with tempfile.TemporaryDirectory() as tempdir:
        config = HierarchicalModelConfig(textline_model_type="bert-layer")
        model1 = SimpleHierarchicalModel(config)
        model1.save_pretrained(f"{tempdir}/tmp-model")

        model2 = SimpleHierarchicalModel.from_pretrained(f"{tempdir}/tmp-model")

        assert compare_models(model1, model2) == 0


def test_load_hierarchical_with_preloaded_weights():

    config = HierarchicalModelConfig(
        textline_model_type="bert-base-uncased",
        load_weights_from_existing_model=True,
    )

    model1 = SimpleHierarchicalModel(config)
    base_bert_model = AutoModel.from_pretrained("bert-base-uncased")

    assert (
        compare_models(model1.textline_encoder.embeddings, base_bert_model.embeddings)
        == 0
    )
    assert (
        compare_models(
            model1.textline_encoder.encoder.layer[0],
            base_bert_model.encoder.layer[0],
        )
        == 0
    )
    assert compare_models(model1.textline_model, base_bert_model) == 0


def test_reload_hierarchical_with_preloaded_weights():
    # Ensures the reloading model weights does not influence the reloading
    # of the models
    with tempfile.TemporaryDirectory() as tempdir:
        config = HierarchicalModelConfig(
            textline_model_type="bert-base-uncased",
            load_weights_from_existing_model=True,
        )

        model1 = SimpleHierarchicalModel(config)
        model1.init_weights()  # changes the current model weights
        model1.save_pretrained(f"{tempdir}/tmp-model")

        model2 = SimpleHierarchicalModel.from_pretrained(f"{tempdir}/tmp-model")
        assert compare_models(model1, model2) == 0


def test_hierarchical_model():
    config = HierarchicalModelConfig(
        num_labels=10,
        textline_model_type="bert-layer",
    )
    model = SimpleHierarchicalModel(config)
    model.eval()

    input_ids = torch.randint(30522, size=(1, 200, 25))
    bbox = create_bbox(1, 200)

    outputs = model(input_ids)
    assert outputs[0].shape == torch.Size([1, 200, config.hidden_size])

    outputs2 = model(input_ids, bbox)
    assert torch.equal(outputs[0], outputs2[0])

    config = HierarchicalModelConfig(
        num_labels=10,
        textline_model_type="layoutlm-layer",
    )
    model = SimpleHierarchicalModel(config)
    model.eval()

    outputs = model(input_ids)
    outputs2 = model(input_ids, bbox)
    assert not torch.equal(outputs[0], outputs2[0])


def test_hierarchical_token_classification_model():
    # Ensures the reloading model weights does not influence the reloading
    # of the models

    config = HierarchicalModelConfig(
        num_labels=10, textline_model_type="bert-layer", hidden_dropout_prob=0
    )
    model = HierarchicalModelForTokenClassification(config)
    model.eval()

    input_ids = torch.randint(30522, size=(1, 200, 25))
    bbox = create_bbox(1, 200)

    outputs = model(input_ids, bbox)
    assert outputs[0].shape == torch.Size([1, 200, 10])

    outputs2 = model(input_ids, bbox)
    assert torch.equal(outputs[0], outputs2[0])

    config = HierarchicalModelConfig(
        num_labels=10,
        textline_model_type="layoutlm-layer",
    )
    model = HierarchicalModelForTokenClassification(config)
    model.eval()

    outputs = model(input_ids)
    outputs2 = model(input_ids, bbox)
    assert not torch.equal(outputs[0], outputs2[0])