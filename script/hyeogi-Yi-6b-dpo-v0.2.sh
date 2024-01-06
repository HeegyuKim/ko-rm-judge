model_id="hyeogi/Yi-6b-dpo-v0.2"
helpful="data/$model_id/gpt4evol.json"
safety="data/$model_id/ko-ethical-questions.json"
batch_size=1

python generate.py \
    --model_id $model_id \
    --testset gpt4evol \
    --batch_size $batch_size \
    --output_filename $helpful \
    --prompt_template vicuna \
    --limit 100 

python generate.py \
    --model_id $model_id \
    --testset ko-ethical-questions \
    --batch_size $batch_size \
    --output_filename $safety \
    --prompt_template vicuna \
    --limit 100 

python eval.py \
    --name helpful \
    --reward_model_id heegyu/ko-reward-model-helpful-1.3b-v0.2 \
    --batch_size $batch_size \
    --filename $helpful

python eval.py \
    --name safety \
    --reward_model_id heegyu/ko-reward-model-safety-1.3b-v0.2 \
    --batch_size $batch_size \
    --filename $safety