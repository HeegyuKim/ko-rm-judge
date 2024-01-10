TEMPLATE_42DOT = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<human>:\n' }}{% elif message['role'] == 'assistant' %}{{ '<bot>:\n' }}{% endif %}{{ message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<bot>:\n' }}{% endif %}"
TEMPLATE_VICUNA = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' }}{% endif %}{{ message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}"
TEMPLATE_LLAMA = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
TEMPLATE_ZEPHYR = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
TEMPLATE_MIDM = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '###User;' }}{% elif message['role'] == 'assistant' %}{{ '###Midm;' }}{% endif %}{{ message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '###Midm;\n' }}{% endif %}"

TEMPLATE_KULLM_NO_INPUT = "{{ '아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n' }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### 명령어:\n' }}{% elif message['role'] == 'assistant' %}{{ '### 응답:\n' }}{% endif %}{{ message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### 응답:\n' }}{% endif %}"
TEMPLATE_KOALPACA = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### 질문:' }}{% elif message['role'] == 'assistant' %}{{ '### 답변:' }}{% endif %}{{ message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### 답변:' }}{% endif %}"
TEMPLATE_CHATML = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}{{ message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant:\n' }}{% endif %}"

PROMPT_TEMPLATES = {
    "42dot": TEMPLATE_42DOT,
    "42dot/42dot_LLM-SFT-1.3B": TEMPLATE_42DOT,
    "heegyu/ko-reward-model-helpful-1.3b-v0.2": TEMPLATE_42DOT,
    "heegyu/ko-reward-model-safety-1.3b-v0.2": TEMPLATE_42DOT,
    
    "vicuna": TEMPLATE_VICUNA,
    "solar": TEMPLATE_VICUNA,
    
    "llama": TEMPLATE_LLAMA,
    
    "zephyr": TEMPLATE_ZEPHYR,
    
    "KT-AI/midm-bitext-S-7B-inst-v1": TEMPLATE_MIDM,
    "midm": TEMPLATE_MIDM,
    
    "kullm-no-input": TEMPLATE_KULLM_NO_INPUT,
    "nlpai-lab/kullm-polyglot-12.8b-v3": TEMPLATE_KULLM_NO_INPUT,
    "nlpai-lab/kullm-polyglot-5.8b-v2": TEMPLATE_KULLM_NO_INPUT,

    "beomi/KoAlpaca-Polyglot-12.8B": TEMPLATE_KOALPACA,
    "beomi/KoAlpaca-Polyglot-5.8B": TEMPLATE_KOALPACA,
    "kfkas/Llama-2-ko-7b-Chat": TEMPLATE_KOALPACA,

    "maywell/Synatra-Yi-Ko-6B": TEMPLATE_CHATML,
}


MTBENCH_PROMPT = {

    "single-v1": "공정한 심사위원으로써 아래에 표시된 사용자의 질문에 대해 AI 어시스턴트가 제공하는 응답의 품질을 평가하세요. 평가는 응답의 유용성, 관련성, 정확성, 깊이, 창의성 및 세부 수준과 같은 요소를 고려해야 합니다. 간단한 설명을 제공하는 것으로 평가를 시작하고 가능한 객관적이어야 합니다. 설명을 제공한 후 [[rating]] 이같은 형식을 엄격하게 따라 응답을 1에서 10까지의 척도로 평가해야 합니다. 척도 예시: \"Rating: [[5]]\"",
    "single-math-v1": "공정한 심사위원으로써 아래에 표시된 사용자 질문에 대해 AI 어시스턴트가 제공하는 응답의 품질을 평가하세요. 평가는 정확성과 유용성을 고려해야 합니다. 정답 답변과 AI 어시스턴트의 답변이 주어집니다. 둘을 비교하여 평가를 시작합니다. 오류를 식별하고 수정합니다. 가능한 객관적이어야 합니다. 설명을 제공한 후 [[rating]] 이같은 형식을 엄격하게 따라 1~10 척도로 응답을 평가해야 합니다. 예를 들어 \"Rating: [[5]]\"",
    "single-v1-multi-turn": "공정한 심사위원으로써 아래에 표시된 사용자의 질문에 대해 AI 어시스턴트가 제공하는 응답의 품질을 평가하세요. 평가는 응답의 유용성, 관련성, 정확성, 깊이, 창의성 및 세부 수준과 같은 요소를 고려해야 합니다. 평가는 두 번째 사용자 질문에 대한 AI 어시스턴트의 대답에 초점을 맞추어야 합니다. 간단한 설명을 제공하는 것으로 평가를 시작하고 가능한 객관적이어야 합니다. 설명을 제공한 후 [[rating]] 이같은 형식을 엄격하게 따라 응답을 1에서 10까지의 척도로 평가해야 합니다. 척도 예시: \"Rating: [[5]]\"",
    "single-math-v1-multi-turn": "공정한 심사위원으로써 아래에 표시된 사용자 질문에 대해 AI 어시스턴트가 제공하는 응답의 품질을 평가하세요. 평가는 정확성과 유용성을 고려해야 합니다. 평가는 두 번째 사용자 질문에 대한 AI 어시스턴트의 대답에 초점을 맞추어야 합니다. 정답 답변과 AI 어시스턴트의 답변이 주어집니다. 둘을 비교하여 평가를 시작합니다. 오류를 식별하고 수정합니다. 가능한 객관적이어야 합니다. 설명을 제공한 후 [[rating]] 이같은 형식을 엄격하게 따라 1~10 척도로 응답을 평가해야 합니다. 예를 들어 \"Rating: [[5]]\"",
}