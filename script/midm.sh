model_id="KT-AI/midm-bitext-S-7B-inst-v1"
helpful="data/$model_id/gpt4evol.json"
safety="data/$model_id/ko-ethical-questions.json"
batch_size=1

# python generate.py \
#     --model_id $model_id \
#     --testset gpt4evol \
#     --batch_size $batch_size \
#     --output_filename $helpful \
#     --trust_remote_code \
#     --limit 100 

# python generate.py \
#     --model_id $model_id \
#     --testset ko-ethical-questions \
#     --batch_size $batch_size \
#     --output_filename $safety \
#     --trust_remote_code \
#     --limit 100 

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