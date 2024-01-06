
export CUDA_VISIBLE_DEVICES=1

model_id="beomi/Yi-Ko-6B"
peft="heegyu/Yi-ko-6B-OKI-v20231124-2e-5"

test() {
    peft_rev=$1
    helpful="data/$peft-$peft_rev/gpt4evol.json"
    safety="data/$peft-$peft_rev/ko-ethical-questions.json"
    batch_size=1

    python generate.py \
        --model_id $model_id \
        --peft_model_id $peft \
        --peft_model_revision $peft_rev \
        --testset gpt4evol \
        --batch_size $batch_size \
        --output_filename $helpful \
        --prompt_template vicuna \
        --limit 100 

    python generate.py \
        --model_id $model_id \
        --peft_model_id $peft \
        --peft_model_revision $peft_rev \
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
}

test "epoch-1"
test "epoch-2"
test "epoch-3"