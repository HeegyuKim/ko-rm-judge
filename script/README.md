# 생성 및 평가 방법
- 각 모델의 답변 프롬프트는 [../prompt_templates.py](../prompt_templates.py)에 정의되어있습니다. 새로 추가하거나 기존 프롬프트 템플릿을 사용할 수 있습니다.
- 기본 최대 생성 토큰수는 512이며, dtype은 float16 입니다.
- 평가할 보상 모델도 동일하게 프롬프트를 적용할 수 있습니다. 

```
model_id="42dot/42dot_LLM-SFT-1.3B"
helpful="data/42dot_LLM-SFT-1.3B/gpt4evol.json"
safety="data/42dot_LLM-SFT-1.3B/ko-ethical-questions.json"
batch_size=4

# helpful 답변 생성
python generate.py \
    --model_id $model_id \
    --testset gpt4evol \        # 평가할 데이터셋
    --batch_size $batch_size \  # 배치크기
    --output_filename $helpful \# 결과가 저장될 파일명
    --limit 100                 # 개수 제한
    --prompt_template 42dot     # 프롬프트 템플릿, 미지정시 model_id에 따라 사전에 지정된 템플릿 사용 (없으면 에러)

# safety 답변 생성
python generate.py \
    --model_id $model_id \
    --testset ko-ethical-questions \
    --batch_size $batch_size \
    --output_filename $safety \
    --limit 100 

# helpful 답변 평가
python eval.py \
    --name helpful \
    --reward_model_id heegyu/ko-reward-model-helpful-1.3b-v0.2 \
    --batch_size $batch_size \
    --filename $helpful

# safety 답변 평가
python eval.py \
    --name safety \
    --reward_model_id heegyu/ko-reward-model-safety-1.3b-v0.2 \
    --batch_size $batch_size \
    --filename $safety
```