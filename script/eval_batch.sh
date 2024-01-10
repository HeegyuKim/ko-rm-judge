batch_size=4
files="data/**/alpaca-eval-ko.json"

python eval.py \
    --name helpful \
    --reward_model_id heegyu/ko-reward-model-helpful-1.3b-v0.2 \
    --filename "$files" \
    --batch_size $batch_size