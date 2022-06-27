dataset_name="$1"
agg_level="$2"
used_token="$3"
model_name="$4"

cd ../tools

n_gpu=4
case $dataset_name in

  grotoap2)
    num_train_epochs=24
    eval_per_epoch=4
    
    if [ "$agg_level" = "block" ]; then
        total_samples=239998
    else
        total_samples=252583
    fi

    eval_steps="$(expr $total_samples / $n_gpu / 40 \* $eval_per_epoch)"
    ;;

  docbank)
    num_train_epochs=6
    eval_per_epoch=2

    if [ "$agg_level" = "block" ]; then
        total_samples=891944
    else
        total_samples=916082
    fi

    eval_steps="$(expr $total_samples / $n_gpu / 40 \* $eval_per_epoch)"
    ;;

  *)
    echo -n "Unkown Dataset"
    exit
    ;;
esac



output_name="ivila-$used_token-$agg_level"

echo "Dataset Name:                            $dataset_name"
echo "Model Name:                              $model_name"
echo "Training Epochs:                         $num_train_epochs"
echo "The group level is:                      $agg_level"
echo "The used layout indicator token is:      $used_token"
echo "Will be evaluated each $eval_steps steps ($eval_per_epoch epochs)"

echo "The results will be saved in '../checkpoints/$dataset_name/$output_name/${model_name//\//-}'"

python train-ivila.py \
  --model_name_or_path "$model_name" \
  --dataset_name "$dataset_name" \
  --preprocessing_num_workers 20 \
  --output_dir "../checkpoints/$dataset_name/$output_name/${model_name//\//-}" \
  --do_train \
  --do_eval \
  --do_predict \
  --save_strategy 'epoch' \
  --metric_for_best_model 'fscore' \
  --evaluation_strategy 'steps' \
  --eval_steps $eval_steps \
  --num_train_epochs $num_train_epochs \
  --save_total_limit 2 \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 40 \
  --warmup_steps 2000 \
  --load_best_model_at_end \
  --added_special_separation_token $used_token \
  --agg_level $agg_level \
  --fp16 