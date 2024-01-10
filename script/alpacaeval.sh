
model_id="42dot/42dot_LLM-SFT-1.3B"
batch_size=4
testset="alpaca-eval-ko"
additional_eos=''
num_gpus=1
tokenizer=""

gen() {
    model_id=$1
    output="data/$model_id/$testset.json"

    python generate.py \
        --model_id $model_id \
        --tokenizer_id "$tokenizer" \
        --testset $testset \
        --batch_size $batch_size \
        --output_filename $output \
        --additional_eos "$additional_eos" \
        --num_gpus $num_gpus \
        --trust_remote_code \
        --top_p 0.95 \
        --top_k 50 
}

#
# ~3B
# gen "42dot/42dot_LLM-SFT-1.3B"

#
# ~10B
batch_size=2
# gen "KT-AI/midm-bitext-S-7B-inst-v1"

# additional_eos='<|im_end|>'
# gen "maywell/Synatra-Yi-Ko-6B"

# additional_eos='###'
# gen "beomi/KoAlpaca-Polyglot-5.8B"
# gen "kfkas/Llama-2-ko-7b-Chat"

# tokenizer="EleutherAI/polyglot-ko-5.8b"
# gen "nlpai-lab/kullm-polyglot-5.8b-v2"

#
# 13B models
batch_size=2
num_gpus=2
additional_eos='###'
tokenizer="EleutherAI/polyglot-ko-12.8b"
gen "nlpai-lab/kullm-polyglot-12.8b-v3"