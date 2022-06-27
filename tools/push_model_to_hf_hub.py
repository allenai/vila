# This script needs to be run with transformer>=4.7 as we need 
# to call the `push_to_hub` API. 

import json
import sys

sys.path.append("..")
from vila import AutoModelForTokenClassification, AutoTokenizer

def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def write_json(data, filename):
    with open(filename, "w") as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="desc")
    parser.add_argument("--label-path", type=str, help="desc")
    parser.add_argument("--repo-name", type=str, help="desc")
    # parser.add_argument("--api-key", type=str, help="desc")

    parser.add_argument("--agg_level", type=str, default=None, help="desc")
    parser.add_argument("--label_all_tokens", type=str, default=None, help="desc")
    parser.add_argument("--group_bbox_agg", type=str, default=None, help="desc")
    parser.add_argument("--added_special_separation_token", type=str, default=None, help="desc")
    args = parser.parse_args()

    print(f"Loading Models from {args.model_path}")
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print("Model and Tokenizer is loaded", model)
    # Step 1: Change the label_map in the config
    
    print("Start to update configs", model.config)
    model_config = model.config
    labels = load_json(args.label_path)

    if hasattr(model_config, "id2label"):
        assert len(labels) == len(model_config.id2label)
        
    model_config.id2label = {int(key):val for key, val in labels.items()}
    model_config.label2id = {val:key for key, val in labels.items()}

    # Step 2: Add vila config to the config

    vila_preprocessor_config = {}
    if args.agg_level is not None:
        vila_preprocessor_config['agg_level'] = args.agg_level
    if args.label_all_tokens is not None:
        vila_preprocessor_config['label_all_tokens'] = args.label_all_tokens
    if args.group_bbox_agg is not None:
        vila_preprocessor_config['group_bbox_agg'] = args.group_bbox_agg
    if args.added_special_separation_token is not None:
        vila_preprocessor_config['added_special_separation_token'] = args.added_special_separation_token
    
    model_config.vila_preprocessor_config = vila_preprocessor_config

    # Step 3: Misc updates of the config 
    model_config.finetuning_task = 'token_classification'
    del model.config._name_or_path
    print("Config is updated", model.config)

    # Step 4: Save the config and try to push Model to repo
    print("Pushing the model to hub", args.repo_name)
    model.push_to_hub(args.repo_name)
    print("Pushing the tokenizer to hub", args.repo_name)
    tokenizer.push_to_hub(args.repo_name)