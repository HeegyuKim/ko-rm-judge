
model_id="42dot/42dot_LLM-SFT-1.3B"
helpful="data/42dot_LLM-SFT-1.3B/gpt4evol.json"
safety="data/42dot_LLM-SFT-1.3B/pku-saferlhf-ko.json"
batch_size=4

python generate.py \
    --model_id $model_id \
    --testset gpt4evol \
    --batch_size $batch_size \
    --output_filename $helpful \
    --limit 100 

python generate.py \
    --model_id $model_id \
    --testset pku-saferlhf-ko \
    --batch_size $batch_size \
    --output_filename $safety \
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