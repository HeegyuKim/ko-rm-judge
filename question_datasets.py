from datasets import load_dataset

def gpt4evol():
    ds = load_dataset("maywell/gpt4_evol_1.3k", split="train")

    def mapper(x):
        return {
            'conversations':[
                {'role': 'user', 'content': x['question']}
                ],
            'response': x['answer']
        }
    ds = ds.map(mapper, remove_columns=ds.column_names)
    return ds


def pku_saferlhf_ko():
    ds = load_dataset("heegyu/PKU-SafeRLHF-ko", split="test")

    def mapper(x):
        return {
            'conversations':[
                {'role': 'user', 'content': x['prompt_k']}
                ],
            'response': x['response_' + str(x['safer_response_id']) + "_ko"]
        }
    
    ds = ds.map(mapper, remove_columns=ds.column_names)
    return ds

def ko_ethical_questions():
    ds = load_dataset("MrBananaHuman/kor_ethical_question_answer", split="train")
    ds = ds.filter(lambda x: x['label'] == 0)

    def mapper(x):
        return {
            'conversations':[
                {'role': 'user', 'content': x['question']}
                ],
            'response': x['answer']
        }
    
    ds = ds.map(mapper, remove_columns=ds.column_names)
    return ds

DATASETS = {
    "gpt4evol": gpt4evol,
    "pku-saferlhf-ko": pku_saferlhf_ko,
    "ko-ethical-questions": ko_ethical_questions
}