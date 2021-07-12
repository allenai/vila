dataset_name="$1"
model_name="$2"
cd ../tools

n_gpu=4
case $dataset_name in

  grotoap2)
    num_train_epochs=24
    eval_per_epoch=4
    total_samples=238023 
    eval_steps="$(expr $total_samples / $n_gpu / 40 \* $eval_per_epoch)"
    ;;

  docbank)
    num_train_epochs=6
    eval_per_epoch=2
    total_samples=881292
    eval_steps="$(expr $total_samples / $n_gpu / 40 \* $eval_per_epoch)"
    ;;

  *)
    echo -n "Unkown Dataset"
    exit
    ;;
esac

echo $eval_steps

output_name='baseline'

echo "Dataset Name:                            $dataset_name"
echo "Model Name:                              $model_name"
echo "Training Epochs:                         $num_train_epochs"
echo "Will be evaluated each $eval_steps steps ($eval_per_epoch epochs)"

echo "The results will be saved in '../checkpoints/$dataset_name/$output_name/${model_name//\//-}'"

python train-baseline.py \
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
  --fp16 