export CUDA_VISIBLE_DEVICES=0

job() {
    helpful="data/$model_id/gpt4evol.json"
    safety="data/$model_id/ko-ethical-questions.json"
    batch_size=1

    python generate.py \
        --model_id $model_id \
        --testset gpt4evol \
        --batch_size $batch_size \
        --output_filename $helpful \
        --additional_eos '<|im_end|>' \
        --num_gpus $num_gpus \
        --limit 100 

    python generate.py \
        --model_id $model_id \
        --testset ko-ethical-questions \
        --batch_size $batch_size \
        --output_filename $safety \
        --additional_eos '<|im_end|>' \
        --num_gpus $num_gpus \
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
}


model_id="maywell/Synatra-Yi-Ko-6B"
num_gpus=1
job