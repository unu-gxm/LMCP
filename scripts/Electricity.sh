export CUDA_VISIBLE_DEVICES=1

model_name=LMCP
seq_len=96

#Ele
for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ \
    --data_path electricity.csv \
    --model_id electricity$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --patch_len 96 \
    --des 'Exp' \
    --d_model 512 \
    --T_model 256 \
    --d_ff 2048 \
    --dropout 0.3 \
    --top_p 0.0 \
    --pos 0 \
    --learning_rate 0.001 \
    --batch_size 6 \
    --train_epochs 10 \
    --itr 1
done