cd ../tools

dataset_name="$1"
model_name="$2"
textline_encoder_output="$3"
group_bbox_agg="$4"
agg_level="$5"


if [ ! -d "../checkpoints/_base_weights/hvila/$model_name" ]; then
   echo "The H-VILA model's base weights is not initialized.
You might want to run the following command to generate them:
    cd ../tools
    python create_hvila_model_base_weights.py
"
   exit 9999
fi

if [ "${textline_encoder_output}" == "" ]; then
    textline_encoder_output="cls"
fi

if [ "${group_bbox_agg}" == "" ]; then
   group_bbox_agg="first"
fi

if [ "${agg_level}" == "" ]; then
   agg_level="row"
fi

if [ "${model_name:0:5}" = "strong" ]; then
    used_batch_size=10
else
    used_batch_size=40
fi

num_gpu="4"
case $dataset_name in

  grotoap2)
    num_train_epochs=24
    eval_per_epoch=4
    if [ "$agg_level" = "block" ]; then
        sample_size=88630
    else
        sample_size=85902
    fi
    eval_steps="$(expr $sample_size / $num_gpu / $used_batch_size \* $eval_per_epoch)"
    ;;

  docbank)
    num_train_epochs=6
    eval_per_epoch=2
    if [ "$agg_level" = "block" ]; then
        sample_size=401627
    else
        sample_size=398365
    fi
    eval_steps="$(expr $sample_size / $num_gpu / $used_batch_size \* $eval_per_epoch)"
    ;;

  *)
    echo -n "Unkown Dataset"
    exit
    ;;
esac

echo "Dataset Name:                            $dataset_name"
echo "Model Name:                              $model_name"
echo "The number of training epochs:           $num_train_epochs"
echo "Textline Encoding Output:                $textline_encoder_output"
echo "How to synthesize grouping bounding box: $group_bbox_agg"
echo "The group level is:                      $agg_level"
echo "The batch size for training:             $used_batch_size"
echo "Will be evaluated each $eval_steps steps ($eval_per_epoch epochs)"

output_name="hvila-$agg_level"
model_save_name="${model_name//\//-}-$textline_encoder_output-$group_bbox_agg"

echo "The results will be saved in '../checkpoints/$dataset_name/$output_name/$model_save_name'"

python train-hvila.py \
  --model_name_or_path "../checkpoints/_base_weights/hvila/$model_name" \
  --output_dir "../checkpoints/$dataset_name/$output_name/$model_save_name" \
  --dataset_name "$dataset_name" \
  --preprocessing_num_workers 20 \
  --do_train \
  --do_eval \
  --do_predict \
  --save_strategy 'epoch' \
  --metric_for_best_model 'fscore' \
  --evaluation_strategy 'steps' \
  --eval_steps $eval_steps \
  --num_train_epochs $num_train_epochs \
  --save_total_limit 2 \
  --per_device_train_batch_size $used_batch_size \
  --per_device_eval_batch_size $used_batch_size \
  --warmup_steps 2000 \
  --agg_level $agg_level \
  --textline_encoder_output $textline_encoder_output \
  --group_bbox_agg $group_bbox_agg \
  --fp16 