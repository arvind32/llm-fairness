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

    df = pd.read_csv("/local/zemel/arvind/code/llm_fairness/data/generated_resumes_with_personas_no_race.csv")
    return df

def load_resume_data_with_race():
    
    resumes_df = pd.read_csv("/local/zemel/arvind/code/llm_fairness/data/generated_resumes_with_personas_no_race.csv")
    print('testing testing')
    # print(len(resumes_df))
    # print(resumes_df.head())
    # print()
    # resumes_df = resumes_df[resumes_df["job"] == args.job]

    full_names_df = pd.read_csv("/local/zemel/arvind/code/llm_fairness/data/generated_names.csv")
    # print(full_names_df.head())
    
    flag = False

    for race in ["white", "black", "asian", "hispanic"]:

        # if race == "anon":

        #     data_df = resumes_df
        #     # resumes = data_df["resume"].tolist()

        # else:

        names_df = full_names_df[full_names_df["Race"] == race]
        temp = pd.merge(resumes_df, names_df, on='person_id')
        if flag:
            data_df = pd.concat([data_df, temp], ignore_index=True)
        else:
            data_df = temp
            flag = True

    return data_df