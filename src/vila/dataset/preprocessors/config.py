from dataclasses import dataclass
import json
import warnings
import os

from transformers import AutoConfig
from vila.dataset import preprocessors


@dataclass
class VILAPreprocessorConfig:

    agg_level: str = "row" #"block", "sentence"
    label_all_tokens: bool = False
    group_bbox_agg: str = "first"
    added_special_separation_token: str = "[BLK]"

    def to_json(self, path: str):
        with open(path, "w") as fp:
            json.dump(vars(self), fp)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        
        config = AutoConfig.from_pretrained(model_path)

        if hasattr(config, "vila_preprocessor_config"):
            data_json = config.vila_preprocessor_config.copy()
            if "added_special_sepration_token" in data_json:
                data_json["added_special_separation_token"] = data_json.pop("added_special_sepration_token")
                # Fix an old typo in the config
            data_json.update(kwargs)
            return cls(**data_json)
            # We store the vila-preprocessor configs inside
            # a typical hf config json.
    
        # If not present, we use the default config
        warnings.warn("The vila_preprocessor_config is not present in the config, using the default one.")
        return cls(**kwargs)