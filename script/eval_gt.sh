export CUDA_VISIBLE_DEVICES=1

batch_size=4

python eval_gt.py \
    --name helpful \
    --testset gpt4evol \
    --reward_model_id "heegyu/ko-reward-model-helpful-1.3b-v0.2" \
    --batch_size $batch_size \
    --limit 100

python eval_gt.py \
    --name safety \
    --testset ko-ethical-questions \
    --reward_model_id "heegyu/ko-reward-model-safety-1.3b-v0.2" \
    --batch_size $batch_size \
    --limit 100