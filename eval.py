import fire
from typing import Optional
import jsonlines
import os
from tqdm import tqdm
import numpy as np
from glob import glob


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
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


def build_model_types(dtype: str, device: str):
    if dtype == "int8":
        return {"load_in_8bit": True, "device_map": "auto"}
    elif dtype == "int4":
        return {"load_in_4bit": True, "device_map": "auto"}
    else:
        return {"torch_dtype": DTYPES[dtype], "device_map": device}


def clean_text(text):
    text = text.replace("<[!newline]>", "\n")  # KT/midm
    return text


@torch.no_grad()
def main(
    name: str,
    reward_model_id: str,
    filename: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 1,
    dtype: str = "float16",
    reward_prompt_template: Optional[str] = None,
    model_revision: Optional[str] = None,
    peft_model_id: Optional[str] = None,
    peft_model_revision: Optional[str] = None,
):
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        print(device)

    tokenizer = AutoTokenizer.from_pretrained(reward_model_id, revision=model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_id, revision=model_revision, **build_model_types(dtype, device)
    ).eval()
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if peft_model_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, peft_model_id, revision=peft_model_revision
        )

    if getattr(tokenizer, "chat_template"):
        tokenizer.chat_template = PROMPT_TEMPLATES[
            reward_prompt_template or reward_model_id
        ]

    model_args = dict(
        model_id=reward_model_id,
        model_revision=model_revision,
        peft_model_id=peft_model_id,
        peft_model_revision=peft_model_revision,
    )

    device = model.device
    files = list(glob(filename, recursive=True))
    print("evaluation targets:", files)
    
    def eval_rewards(filename):
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        reward_output_filename = os.path.join(
            dirname, os.path.splitext(basename)[0] + f"_{name}.json"
        )

        with jsonlines.open(filename) as fin:
            dataset = list(fin)
            dataset_len = len(dataset)

            if len(dataset) == 0 or "score" in dataset[0]:
                print(f"pass {filename}")
                return

        all_scores = []

        if os.path.exists(reward_output_filename):
            with jsonlines.open(reward_output_filename) as fin:
                evaluated_items = list(fin)
                skip_lines = len(evaluated_items)
                all_scores = [x["score"] for x in evaluated_items]
                if skip_lines >= dataset_len:
                    print(f"pass {filename}")
                    return
                print(f"파일이 이미 존재하며 {skip_lines}개가 이미 평가되어있습니다.")
        else:
            skip_lines = 0
            all_scores = []

        with jsonlines.open(reward_output_filename, "a") as fout:
            progress = tqdm(total=skip_lines // batch_size, desc=filename)
            for i in range(0, len(dataset), batch_size):
                if i < skip_lines:
                    continue

                # 우선 아이템들을 인코딩합니다.
                end_i = min(dataset_len, i + batch_size)
                input_ids = []
                for j in range(i, end_i):
                    item = dataset[j]
                    conv = item["conversations"] + [
                        {"role": "assistant", "content": clean_text(item["response"])}
                    ]
                    input_id = tokenizer.apply_chat_template(conv)

                    if input_id[-1] != tokenizer.eos_token_id:
                        input_id.append(tokenizer.eos_token_id)

                    input_ids.append(input_id)

                # 점수 측정
                inputs = collator(input_ids)
                inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
                scores = model(**inputs).logits.cpu()[:, 0].tolist()

                # 결과 저장
                for j in range(i, end_i):
                    item = dataset[j]
                    item["score"] = scores[j - i]
                    item["reward_model_args"] = model_args
                    fout.write(item)

                all_scores.extend(scores)
                progress.update(batch_size)

        print(filename)
        print("average score:", np.mean(all_scores))
        print("average std:", np.std(all_scores))


    for file in files:
        eval_rewards(file)

if __name__ == "__main__":
    fire.Fire(main)
