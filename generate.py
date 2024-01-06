import fire
from typing import Optional
import jsonlines
import os
from tqdm import tqdm


from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from prompt_templates import PROMPT_TEMPLATES
from question_datasets import DATASETS
from copy import deepcopy

import torch


DTYPES = {
    'float16': torch.float16,
    'bfloat16': torch.float16,
    'float32': torch.float32,
}

def build_model_types(dtype: str, device: str):
   if dtype == "int8":
      return {
         "load_in_8bit": True,
         "device_map": "auto"
      }
   elif dtype == "int4":
      return {
         "load_in_4bit": True,
         "device_map": "auto"
      }
   else:
      return {
         'torch_dtype': DTYPES[dtype],
         'device_map': device
      }

@torch.no_grad()
def main(
    model_id: str,
    testset: str,
    output_filename: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 1,
    dtype: str = "float16",
    limit: Optional[int] = None,
    prompt_template: Optional[str] = None,
    model_revision: Optional[str] = None,
    peft_model_id: Optional[str] = None,
    peft_model_revision: Optional[str] = None,

    do_sample=True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_new_tokens: Optional[int] = 512
    ):

    if output_filename is None:
       output_filename = f"data/{testset}-" + model_id.replace("/", "__") + ".json"

    if os.path.exists(output_filename):
       with jsonlines.open(output_filename) as fin:
          skip_lines = len(list(fin))
          print(f"파일이 이미 존재하며 {skip_lines}개가 이미 생성되어있습니다.")
    else:
       skip_lines = 0

    if device == "auto":
       device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision, **build_model_types(dtype, device)).eval()
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if peft_model_id:
       from peft import PeftModel
       model = PeftModel.from_pretrained(model, peft_model_id, revision=peft_model_revision)


    if getattr(tokenizer, "default_chat_template"):
        tokenizer.chat_template = PROMPT_TEMPLATES[prompt_template or model_id]

    
    gen_args = dict(
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    model_args = dict(
        model_id=model_id,
        model_revision=model_revision,
        peft_model_id=peft_model_id,
        peft_model_revision=peft_model_revision
    )

    dirname = os.path.dirname(output_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    device = model.device
    dataset = DATASETS[testset]()
    with jsonlines.open(output_filename, "a") as fout:
        dataset_len = len(dataset)
        progress = tqdm(total=limit or len(dataset) // batch_size)
        for i in range(0, len(dataset), batch_size):
            if i < skip_lines:
                continue
            
            if i >= limit:
                print(f"{limit} 제한으로 중단합니다.")
                break
          
            # 우선 아이템들을 인코딩합니다.
            end_i = min(dataset_len, i + batch_size)
            input_ids = []
            for j in range(i, end_i):
                item = dataset[j]
                input_id = tokenizer.apply_chat_template(item["conversations"], add_generation_prompt=True)
                input_ids.append(input_id)

            # 점수 측정
            inputs = collator(input_ids)
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}

            prompt_len = inputs['input_ids'].shape[1]
            responses = model.generate(**inputs, **gen_args).cpu()
            # print(tokenizer.batch_decode(responses, skip_special_tokens=True))
            responses = responses[:, prompt_len:]
            responses = tokenizer.batch_decode(responses, skip_special_tokens=True)

            # 결과 저장
            for j in range(i, end_i):
                item = dataset[j]
                item['response'] = responses[j - i]
                item['model_args'] = model_args
                item['gen_args'] = gen_args
                fout.write(item)

            progress.update(batch_size)
          
    

if __name__ == '__main__':
    fire.Fire(main)