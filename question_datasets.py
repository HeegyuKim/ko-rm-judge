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
                {'role': 'user', 'content': x['prompt']}
                ],
            'response': x['response_' + x['safer_response_id']]
        }
    
    ds = ds.map(mapper, remove_columns=ds.column_names)
    return ds


DATASETS = {
    "gpt4evol": gpt4evol,
    "pku-saferlhf-ko": pku_saferlhf_ko
}