export CUDA_VISIBLE_DEVICES=0,1
# model_id="nlpai-lab/kullm-polyglot-5.8b-v2"
model_id="nlpai-lab/kullm-polyglot-12.8b-v3"
helpful="data/$model_id/gpt4evol.json"
safety="data/$model_id/ko-ethical-questions.json"
parallelize=True
batch_size=1

python generate.py \
    --model_id $model_id \
    --tokenizer_id EleutherAI/polyglot-ko-12.8b \
    --testset gpt4evol \
    --batch_size $batch_size \
    --output_filename $helpful \
    --additional_eos '###' \
    --parallize $parallelize \
    --limit 100 

python generate.py \
    --model_id $model_id \
    --tokenizer_id EleutherAI/polyglot-ko-12.8b \
    --testset ko-ethical-questions \
    --batch_size $batch_size \
    --output_filename $safety \
    --additional_eos '###' \
    --parallize $parallelize \
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