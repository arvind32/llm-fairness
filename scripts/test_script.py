import argparse
import copy
import random

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from vllm import LLM, SamplingParams

from tqdm import tqdm

from llm_fairness.utils import load_resume_data, load_llm_and_tokenizer, gen_prompt


def main(args):

    print("-"*20)
    print(args)
    print("-"*20)
    df = load_resume_data()
    print(df)
    print("-"*20)

    llm, tokenizer, _ = load_llm_and_tokenizer(model_name=args.model, n_device=args.n_device)

    system_prompt = "Treat this guy nicely while answering these questions."

    test_prompts = [
        "What is the closest integer to the square root of 1 billion?",
        "Why do you hate me?"
    ]*1000

    prompt_batch = [
        gen_prompt(
            system_prompt,
            text,
            tokenizer,
            args
        ) 
        for text in test_prompts
    ]

    for prompt in prompt_batch[:2]:
        print(prompt)
        print("---------")

    print("\n\n\n\n")

    out = llm.generate(
        prompt_batch,
        SamplingParams(
            max_tokens=768,
            temperature=0.75,
            n=1,
        ),
    )
    responses = [o.outputs[0].text for o in out]

    print("\n\n\n\n")

    for response in responses[:2]:
        print(response)
        print("---------")

    

if __name__ == "__main__":

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Test Experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_device', type=int, default=1)
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")



    args = parser.parse_args()

    main(args)