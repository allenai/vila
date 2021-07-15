import sys

sys.path.append("../")
from vila.models.configuration_hierarchical_model import HierarchicalModelConfig
from vila.models.hierarchical_model import SimpleHierarchicalModel

import argparse


MODEL_NAME_CONFIG_PAIR = {
    "weak-weak-bert": dict(
        textline_encoder_type="bert-layer",
        textline_model_type="bert-layer",
    ),
    "weak-strong-bert": dict(
        textline_encoder_type="bert-layer",
        textline_model_type="bert-base-uncased",
    ),
    "strong-weak-bert": dict(
        textline_encoder_type="bert-base-uncased",
        textline_model_type="bert-layer",
    ),
    "strong-strong-bert": dict(
        textline_encoder_type="bert-base-uncased",
        textline_model_type="bert-base-uncased",
    ),
    "weak-weak-layoutlm": dict(
        textline_encoder_type="bert-layer",
        textline_model_type="layoutlm-layer",
    ),
    "weak-strong-layoutlm": dict(
        textline_encoder_type="bert-layer",
        textline_model_type="layoutlm-base-uncased",
    ),
    "strong-weak-layoutlm": dict(
        textline_encoder_type="bert-base-uncased",
        textline_model_type="layoutlm-layer",
    ),
    "strong-strong-layoutlm": dict(
        textline_encoder_type="bert-base-uncased",
        textline_model_type="layoutlm-base-uncased",
    ),
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="../checkpoints/_base_weights/hvila",
        help="The path to store the pre-trained weights",
    )
    args = parser.parse_args()

    # Weak-Weak-BERT
    for model_name, model_configs in MODEL_NAME_CONFIG_PAIR.items():

        print(f"Creating the {model_name}")

        config = HierarchicalModelConfig(
            load_weights_from_existing_model=True, **model_configs
        )
        model = SimpleHierarchicalModel(config)
        model.config.load_weights_from_existing_model = False
        model.save_pretrained(f"{args.save_path}/{model_name}")