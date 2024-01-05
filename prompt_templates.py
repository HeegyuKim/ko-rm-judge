

TEMPLATE_42DOT = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<human>:\n' }}{% elif message['role'] == 'assistant' %}{{ '<bot>:\n' }}{% endif %}{{ message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<bot>:\n' }}{% endif %}"

PROMPT_TEMPLATES = {
    "42dot": TEMPLATE_42DOT,
    "42dot/42dot_LLM-SFT-1.3B": TEMPLATE_42DOT,
    "heegyu/ko-reward-model-helpful-1.3b-v0.2": TEMPLATE_42DOT,
    "heegyu/ko-reward-model-safety-1.3b-v0.2": TEMPLATE_42DOT,
}