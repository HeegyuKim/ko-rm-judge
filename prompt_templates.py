TEMPLATE_42DOT = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<human>:\n' }}{% elif message['role'] == 'assistant' %}{{ '<bot>:\n' }}{% endif %}{{ message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<bot>:\n' }}{% endif %}"
TEMPLATE_VICUNA = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' }}{% endif %}{{ message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}"
TEMPLATE_LLAMA = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
TEMPLATE_ZEPHYR = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
TEMPLATE_MIDM = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '###User;' }}{% elif message['role'] == 'assistant' %}{{ '###Midm;' }}{% endif %}{{ message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '###Midm;\n' }}{% endif %}"

TEMPLATE_KULLM_NO_INPUT = "{{ '아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n' }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### 명령어:\n' }}{% elif message['role'] == 'assistant' %}{{ '### 응답:\n' }}{% endif %}{{ message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### 응답:\n' }}{% endif %}"

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
}
