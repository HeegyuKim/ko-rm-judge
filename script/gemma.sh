
run_eval() {
    size=$1 # "2b-it"
    dtype="bfloat16"

    model_id="google/gemma-$size"
    helpful="data/gemma-$size/alpaca-eval-ko.json"
    safety="data/gemma-$size/ko-ethical-questions.json"
    batch_size=4

    python generate.py \
        --model_id $model_id \
        --testset alpaca-eval-ko \
        --dtype $dtype \
        --batch_size $batch_size \
        --prompt_template $model_id \
        --output_filename $helpful \

    python generate.py \
        --model_id $model_id \
        --testset ko-ethical-questions \
        --batch_size $batch_size \
        --dtype $dtype \
        --prompt_template $model_id \
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

# run_eval "2b-it"
# run_eval "2b-it"
run_eval "$1"