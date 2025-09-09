import pandas as pd
from vllm import LLM, SamplingParams
from pathlib import Path

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

def load_resume_data_with_race(overwrite = False):
    
    folder = Path("/local/zemel/arvind/code/llm_fairness/data")
    filename = "generated_resumes_with_namesinresume.csv"

    # if file already exists, return the corresponding dataframe without overwriting
    if (folder / filename).exists() and not overwrite:
        resumes_df = pd.read_csv(folder / filename)
        return resumes_df

    # if file does not exist, or if overwrite is True, create a new file
    noname_resumes_df = pd.read_csv("/local/zemel/arvind/code/llm_fairness/data/generated_resumes_with_personas_no_race.csv")

    full_names_df = pd.read_csv("/local/zemel/arvind/code/llm_fairness/data/generated_names.csv")
    
    flag = False

    for race in ["white", "black", "asian", "hispanic"]:

        names_df = full_names_df[full_names_df["Race"] == race]
        temp = pd.merge(noname_resumes_df, names_df, on='person_id')
        if flag:
            resumes_df = pd.concat([resumes_df, temp], ignore_index=True)
        else:
            resumes_df = temp
            flag = True

    # Replace placeholders in 'resume' field with actual names and emails
    resumes_df['resume'] = resumes_df.apply(lambda row: row['resume'].replace('[NAME]', row['Name']), axis=1)
    resumes_df['resume'] = resumes_df.apply(lambda row: row['resume'].replace('[EMAIL]', row['Email']), axis=1)
    
    resumes_df.to_csv('/local/zemel/arvind/code/llm_fairness/data/generated_resumes_with_namesinresume.csv', 
                      index=False)
    
    return resumes_df