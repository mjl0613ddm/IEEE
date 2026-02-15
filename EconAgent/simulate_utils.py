import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy


save_path = './'

DEFAULT_LLM_CONFIG = {
    "provider": "openai_proxy",
    "api_base": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key_env": "OPENAI_API_KEY",
    "api_key": None, 
}
LLM_CONFIG = DEFAULT_LLM_CONFIG.copy()

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

# Pricing: $0.5820 / 1M input tokens, $0.5820 / 1M output tokens
prompt_cost_1k = 0.5820 / 1000  # per 1K tokens
completion_cost_1k = 0.5820 / 1000  # per 1K tokens


def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100):
    import sys
    from functools import partial
    get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    total_cost = sum([cost for _, cost in results])
    return [response for response, _ in results], total_cost

def get_completion(dialogs, temperature=0, max_tokens=100):
    import openai
    import time
    import sys
    import os

    api_key = LLM_CONFIG.get("api_key") or os.getenv(LLM_CONFIG.get("api_key_env", ""))
    if not api_key:
        raise RuntimeError("Remote LLM provider requires an API key. Set it via config or environment.")

    openai.api_key = api_key
    api_base = LLM_CONFIG.get("api_base")
    if api_base:
        openai.api_base = api_base
    model = LLM_CONFIG.get("model", "gpt-4o")
    
    max_retries = 20
    for i in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=dialogs,
                temperature=temperature,
                max_tokens=max_tokens
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            this_cost = prompt_tokens/1000*prompt_cost_1k + completion_tokens/1000*completion_cost_1k
            return response.choices[0].message["content"], this_cost
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(6)
            else:
                return "Error", 0.0

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

def configure_simulation(save_path_override=None, llm_overrides=None):
    """
    Update module-level configuration for output paths and language model access.
    """
    global save_path, LLM_CONFIG

    if save_path_override:
        save_path = os.path.abspath(save_path_override)
    else:
        save_path = os.path.abspath(save_path) if not os.path.isabs(save_path) else save_path

    overrides = llm_overrides or {}
    for key, value in overrides.items():
        if value is None:
            continue
        LLM_CONFIG[key] = value
