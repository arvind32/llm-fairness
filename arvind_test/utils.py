import pandas as pd
from vllm import LLM, SamplingParams


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def gen_prompt(system_prompt, text, tokenizer, args):
    if "gemma" in args.model:
        messages = [
            {"role": "user", "content": f"{system_prompt} Here is a user query: {text}"},
        ]

    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,

    )
    return prompt


def load_llm_and_tokenizer(model_name, n_device):

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=n_device,
        disable_log_stats=True,
        download_dir="/local/zemel/hf/"
    )
    tokenizer = llm.get_tokenizer()
    if model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "meta-llama/Meta-Llama-3-70B-Instruct":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]
    return llm, tokenizer, terminators


def load_resume_data():

    df = pd.read_csv("/local/zemel/tom/code/arvind_test/data/generated_resumes_with_personas_no_race.csv")
    return df
