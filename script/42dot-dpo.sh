export CUDA_VISIBLE_DEVICES=1

model_id="42dot/42dot_LLM-SFT-1.3B"
peft="heegyu/42dot-SFT-DPO-v0.1"

test() {
    peft_rev=$1
    helpful="data/$peft-$peft_rev/gpt4evol.json"
    safety="data/$peft-$peft_rev/ko-ethical-questions.json"
    batch_size=4

    python generate.py \
        --model_id $model_id \
        --peft_model_id $peft \
        --peft_model_revision $peft_rev \
        --testset gpt4evol \
        --batch_size $batch_size \
        --output_filename $helpful \
        --limit 100 

    python generate.py \
        --model_id $model_id \
        --peft_model_id $peft \
        --peft_model_revision $peft_rev \
        --testset ko-ethical-questions \
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
}

test "steps-63332"
# test "steps-126664"
test "steps-189996"