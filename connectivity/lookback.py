import numpy as np
import torch

EPSILON = 1e-8

def compute_lookback_lens(matrix, **params):
    if not "prompt_length" in params:
        raise Exception("Not found prompt_length in metadata")

    prompt_length = params["prompt_length"]

    # matrix dimension is S, S
    generated_attnetion = matrix[prompt_length:, :]
    attn_prompt = generated_attnetion[:, :prompt_length].mean(axis=-1)
    attn_answer = generated_attnetion[:, prompt_length:].mean(axis=-1)

    token_span_lookback_ratio = attn_prompt/(attn_prompt+attn_answer)

    return {
        "lookback": token_span_lookback_ratio.mean(),
    }


if __name__ == "__main__":
    attention = torch.ones(24, 24, 100, 100)

    # for single head
    matrix = attention[0, 0]   # shape (100, 100)
    params = {
        "prompt_length": 40,
    }

    print(compute_lookback_lens(matrix=matrix, **params))
