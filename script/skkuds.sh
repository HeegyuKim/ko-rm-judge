

model_id="logicker/SkkuDataScience-10.7B-v5"
helpful="data/logicker/SkkuDataScience-10.7B-v5/gpt4evol.json"
safety="data/logicker/SkkuDataScience-10.7B-v5/ko-ethical-questions.json"
batch_size=1

python generate.py \
    --model_id $model_id \
    --testset gpt4evol \
    --batch_size $batch_size \
    --output_filename $helpful \
    --prompt_template zephyr \
    --limit 100 

python generate.py \
    --model_id $model_id \
    --testset ko-ethical-questions \
    --batch_size $batch_size \
    --output_filename $safety \
    --prompt_template zephyr \
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