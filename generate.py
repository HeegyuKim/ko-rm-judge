import fire
from typing import Optional, List
import jsonlines
import os
from tqdm import tqdm


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AutoConfig,
    StoppingCriteria,
)
from prompt_templates import PROMPT_TEMPLATES
from question_datasets import DATASETS
from copy import deepcopy

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.float16,
    "float32": torch.float32,
}


def build_model_types(dtype: str, device: str, num_gpus: int):
    if num_gpus > 1:
        return {"torch_dtype": DTYPES[dtype]}
    elif dtype == "int8":
        return {"load_in_8bit": True, "device_map": "auto"}
    elif dtype == "int4":
        return {"load_in_4bit": True, "device_map": "auto"}
    else:
        return {"torch_dtype": DTYPES[dtype], "device_map": device}

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, eos_list: List[str]):
        self.eos_list = [tokenizer.encode(x) for x in eos_list]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for eos_sequence in self.eos_list:
            last_ids = input_ids[:,-len(eos_sequence):].tolist()
            if eos_sequence in last_ids:
                return True
        return False

@torch.no_grad()
def main(
    model_id: str,
    testset: str,
    output_filename: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 1,
    dtype: str = "float16",
    limit: Optional[int] = -1,
    prompt_template: Optional[str] = None,
    model_revision: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    peft_model_id: Optional[str] = None,
    peft_model_revision: Optional[str] = None,
    trust_remote_code: bool = False,
    num_gpus: int = 1,
    do_sample=True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_new_tokens: Optional[int] = 512,
    additional_eos: Optional[str] = None,
    print_generation: bool = True
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

    if tokenizer_id is None or len(tokenizer_id.strip()) == 0:
        tokenizer_id = model_id

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=model_revision,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        **build_model_types(dtype, device, num_gpus),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True, offload_state_dict=True
    ).eval()
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if num_gpus > 1:
        import tensor_parallel as tp
        model = tp.tensor_parallel(model, [f"cuda:{i}" for i in range(num_gpus)])


    if peft_model_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, peft_model_id, revision=peft_model_revision
        )

    if prompt_template:
        tokenizer.chat_template = PROMPT_TEMPLATES[prompt_template]
    elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        pass
    else:
        tokenizer.chat_template = PROMPT_TEMPLATES[model_id]

    gen_args = dict(
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if additional_eos:
        gen_args["stopping_criteria"] = [EosListStoppingCriteria(tokenizer, [additional_eos])]

    model_args = dict(
        model_id=model_id,
        model_revision=model_revision,
        peft_model_id=peft_model_id,
        peft_model_revision=peft_model_revision,
        dtype=dtype,
    )

    dirname = os.path.dirname(output_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    device = model.device
    dataset = DATASETS[testset]()
    gen_args_save = gen_args.copy()
    if additional_eos:
        gen_args_save.pop("stopping_criteria")

    with jsonlines.open(output_filename, "a") as fout:
        dataset_len = len(dataset)
        progress = tqdm(total=limit if limit > 0 else len(dataset))
        for i in range(0, len(dataset), batch_size):
            if i < skip_lines:
                continue

            if limit > 0 and i >= limit:
                print(f"{limit} 제한으로 중단합니다.")
                break

            # 우선 아이템들을 인코딩합니다.
            end_i = min(dataset_len, i + batch_size)
            input_ids = []
            for j in range(i, end_i):
                item = dataset[j]
                input_id = tokenizer.apply_chat_template(
                    item["conversations"], add_generation_prompt=True
                )
                input_ids.append(input_id)

            # 점수 측정
            inputs = collator(input_ids)
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}

            prompt_len = inputs["input_ids"].shape[1]
            responses = model.generate(**inputs, **gen_args).cpu()
            if print_generation:
                for full_output in tokenizer.batch_decode(responses, skip_special_tokens=True):
                    print(full_output)

            responses = responses[:, prompt_len:]
            responses = tokenizer.batch_decode(responses, skip_special_tokens=True)

            # 결과 저장
            for j in range(i, end_i):
                r = responses[j - i]
                item = dataset[j]

                if additional_eos and additional_eos in r:
                    r = r.split(additional_eos, 1)[0]
                    
                item["response"] = r
                item["model_args"] = model_args
                item["gen_args"] = gen_args_save
                fout.write(item)

            progress.update(batch_size)


if __name__ == "__main__":
    fire.Fire(main)
