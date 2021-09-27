cd ../tools

BASE_PATH="../"

# DocBank IVILA Block Finetuned 
python push_model_to_hf_hub.py \
    --model-path $BASE_PATH/checkpoints/docbank/layout_indicator-BLK-block/microsoft-layoutlm-base-uncased/ \
    --label-path $BASE_PATH/data/docbank/grouped-v6/labels.json \
    --repo-name ivila-block-layoutlm-finetuned-docbank \
    --agg_level "block" \
    --group_bbox_agg "first" \
    --added_special_sepration_token "[BLK]"

# DocBank HVILA Block Finetuned 
python push_model_to_hf_hub.py \
    --model-path $BASE_PATH/checkpoints/docbank/hierarchical_model-block/weak-strong-layoutlm-average-first/ \
    --label-path $BASE_PATH/data/docbank/grouped-v6/labels.json \
    --repo-name hvila-block-layoutlm-finetuned-docbank \
    --agg_level "block" \
    --group_bbox_agg "first" \

# DocBank HVILA Row Finetuned 
python push_model_to_hf_hub.py \
    --model-path $BASE_PATH/checkpoints/docbank/hierarchical_model-row/weak-strong-layoutlm-average-first/ \
    --label-path $BASE_PATH/data/docbank/grouped-v6/labels.json \
    --repo-name hvila-row-layoutlm-finetuned-docbank \
    --agg_level "row" \
    --group_bbox_agg "first" \

# GROTOAP2 IVILA Block Finetuned 
python push_model_to_hf_hub.py \
    --model-path $BASE_PATH/checkpoints/grotoap/layout_indicator-BLK-block/microsoft-layoutlm-base-uncased/ \
    --label-path $BASE_PATH/data/docbank/grouped-v6/labels.json \
    --repo-name ivila-block-layoutlm-finetuned-grotoap2 \
    --agg_level "block" \
    --group_bbox_agg "first" \
    --added_special_sepration_token "[BLK]"

# GROTOAP2 HVILA Block Finetuned 
python push_model_to_hf_hub.py \
    --model-path $BASE_PATH/checkpoints/grotoap/hierarchical_model-block/weak-strong-layoutlm-average-first/ \
    --label-path $BASE_PATH/data/grotoap2/grouped-v2/labels.json    \
    --repo-name hvila-block-layoutlm-finetuned-grotoap2 \
    --agg_level "block" \
    --group_bbox_agg "first" \

# GROTOAP2 HVILA Row Finetuned 
python push_model_to_hf_hub.py \
    --model-path $BASE_PATH/checkpoints/grotoap/hierarchical_model-row/weak-strong-layoutlm-average-first/ \
    --label-path $BASE_PATH/data/grotoap2/grouped-v2/labels.json    \
    --repo-name hvila-row-layoutlm-finetuned-grotoap2 \
    --agg_level "row" \
    --group_bbox_agg "first" \
