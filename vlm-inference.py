from dotenv import load_dotenv

load_dotenv(override=True)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from magic.save import save_results
from magic.utils import all_seed, clear_cuda, load_image

clear_cuda()
all_seed()

BASE_SAVE_PATH = Path(os.getenv("BASE_SAVE_DIR"))


def load_models(model_name):
    if model_name == "llava-med-v1.5-mistral-7b":
        from llava.mm_utils import (
            get_model_name_from_path,
        )
        from llava.model.builder import load_pretrained_model

        model_name = get_model_name_from_path("microsoft/llava-med-v1.5-mistral-7b")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path="microsoft/llava-med-v1.5-mistral-7b",
            model_base=None,
            model_name=model_name,
            load_4bit=False,
        )
        return model, tokenizer, image_processor, context_len
    if model_name == "medgemma-4b-it":
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        return model, processor, None, None


def process_image_for_llava(image, image_processor):
    from llava.mm_utils import (
        process_images,
    )

    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)


def create_prompt(prompt: str):
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
    )
    from llava.conversation import conv_templates

    CONV_MODE = "llava_v0"
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv


def get_token_type_indices(input_ids, generated_ids, attention_seq_len, image_token_id=-200):
    """
    Get indices for visual, text, and answer tokens in the attention matrix space.

    Args:
        input_ids: Original input with -200 placeholders
        generated_ids: Full sequence including generated tokens (with -200 placeholders)
        attention_seq_len: Actual sequence length from attention matrix
        image_token_id: Placeholder token ID (default: -200)

    Returns:
        Dictionary with visual, text, and answer token indices in attention space
    """
    placeholder_mask = input_ids == image_token_id
    placeholder_positions = placeholder_mask.nonzero(as_tuple=True)[1]
    num_placeholders = len(placeholder_positions)
    input_seq_len = input_ids.shape[1]

    num_generated_tokens = generated_ids.shape[1] - input_ids.shape[1]

    attention_seq_len_without_generated = attention_seq_len - num_generated_tokens
    total_image_tokens = attention_seq_len_without_generated - input_seq_len + num_placeholders
    tokens_per_image = total_image_tokens // num_placeholders
    visual_token_indices = []
    offset = 0
    text_token_indices = []

    for input_idx in range(input_seq_len):
        if input_idx in placeholder_positions:
            image_start = input_idx + offset
            image_end = image_start + tokens_per_image
            visual_token_indices.extend(range(image_start, image_end))
            offset += tokens_per_image - 1
        else:
            attention_idx = input_idx + offset
            text_token_indices.append(attention_idx)

    initial_len_in_attention = input_seq_len + offset
    answer_token_indices = list(range(initial_len_in_attention, attention_seq_len))

    return {
        "visual_token_indices": visual_token_indices,
        "text_token_indices": text_token_indices,
        "answer_token_indices": answer_token_indices,
        "tokens_per_image": tokens_per_image,
    }


def process_weights(model, processor, image_processor, context_len, image, question):
    if model_name == "llava-med-v1.5-mistral-7b":
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
        )
        from llava.mm_utils import (
            tokenizer_image_token,
        )

        image_tensor = process_image_for_llava(image, image_processor)
        prompt, _ = create_prompt(question)
        input_ids = (
            tokenizer_image_token(prompt, processor, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        max_new_tokens = 256
        image_token_id = -200
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = model(
                    generated_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    use_cache=True,
                )
                next_token_logits = output.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if next_token_id.item() == processor.eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        prompt_len = input_ids.shape[-1]
        answer_ids = generated_ids[0][prompt_len:]
        print(processor.decode(answer_ids))

        attention = output.attentions
        attention_seq_len = output.attentions[0].shape[-1]
        indices = get_token_type_indices(
            input_ids=input_ids, generated_ids=generated_ids, attention_seq_len=attention_seq_len
        )

        answer_token_indices = indices["answer_token_indices"]
        text_token_indices = indices["text_token_indices"]
        visual_token_indices = indices["visual_token_indices"]
        layers = len(output.attentions)
        num_heads = output.attentions[0].shape[1]
        visual_subgraphs = {}
        text_subgraphs = {}
        for layer_idx in range(layers):
            layer_attn = (
                output.attentions[layer_idx][0].detach().cpu().numpy()
            )  # shape (num_heads, seq_len, seq_len)
            visual_subgraphs[layer_idx] = []
            text_subgraphs[layer_idx] = []

            for head_idx in range(num_heads):
                attn = layer_attn[head_idx]
                visual_submat = attn[np.ix_(answer_token_indices, visual_token_indices)]
                text_submat = attn[np.ix_(answer_token_indices, text_token_indices)]
                visual_subgraphs[layer_idx].append(visual_submat)
                text_subgraphs[layer_idx].append(text_submat)

        # for the embedding of each token
        token_embeddings = torch.stack(output.hidden_states, dim=1)[0].detach().cpu().numpy()

        attention = [attn[0].cpu().numpy() for attn in output.attentions]
        prompt_len = input_ids.shape[-1]
        answer_ids = generated_ids[0][prompt_len:]
        answer = processor.decode(answer_ids, skip_special_tokens=True).strip()

        return (
            answer,
            attention,
            visual_subgraphs,
            text_subgraphs,
            visual_token_indices,
            text_token_indices,
            answer_token_indices,
            token_embeddings,
            None,
        )

    if model_name == "medgemma-4b-it":
        prompt = "Answer the question as concisely as possible: " + question
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=200, do_sample=False, use_cache=True
            )
        answer_ids = generated_ids[0][prompt_len:]
        full_sequence = generated_ids
        with torch.no_grad():
            outputs = model(
                input_ids=full_sequence,
                pixel_values=inputs.get("pixel_values"),
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )

        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        input_ids_list = full_sequence[0].cpu().tolist()
        visual_token_indices = [i for i, tid in enumerate(input_ids_list) if tid == image_token_id]
        text_token_indices = [i for i in range(prompt_len) if i not in visual_token_indices]
        answer_token_indices = list(range(prompt_len, len(input_ids_list)))
        layers = len(outputs.attentions)
        num_heads = outputs.attentions[0].shape[1]
        visual_subgraphs = {}
        text_subgraphs = {}
        layer_types = model.config.text_config.layer_types
        global_layer_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        print(f"Extracted Global Layers at: {global_layer_indices}")

        global_attentions = []
        for layer_idx in global_layer_indices:
            layer_attn = outputs.attentions[layer_idx][0].detach().cpu().to(torch.float32).numpy()
            global_attentions.append(layer_attn)
            visual_subgraphs[layer_idx] = []
            text_subgraphs[layer_idx] = []

            for head_idx in range(num_heads):
                attn = layer_attn[head_idx]
                visual_submat = attn[np.ix_(answer_token_indices, visual_token_indices)]
                text_submat = attn[np.ix_(answer_token_indices, text_token_indices)]
                visual_subgraphs[layer_idx].append(visual_submat)
                text_subgraphs[layer_idx].append(text_submat)
        answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
        sequence = generated_ids[0].detach().cpu().numpy()
        return (
            answer,
            global_attentions,
            visual_subgraphs,
            text_subgraphs,
            visual_token_indices,
            text_token_indices,
            answer_token_indices,
            sequence,
        )


def process_dataset(
    model_name,
    model,
    processor,
    image_processor=None,
    context_len=None,
    dataset="flaviagiammarino/vqa-rad",
    split="train",
    from_index=0,
    to_index=int(1e12),
):
    """
     Process samples from  dataset
    Args:
        model: Vision-language model
        processor: Model processor
        split: One of 'random', 'popular', 'adversarial'
        num_samples: Number of samples to process
    """
    import time

    save_folder = (
        BASE_SAVE_PATH / model_name / split / f"{dataset.split('/')[-1]}-{from_index}-{to_index}"
    )
    os.makedirs(save_folder, exist_ok=True)
    dataset = load_dataset(dataset, split=split)
    n_dataset = len(dataset)
    # which ever is the smallest
    to_index = min(to_index, n_dataset - 1)

    results = []

    for idx in range(from_index, to_index + 1):
        start = time.perf_counter()
        sample = dataset[idx]
        image = sample["image"]
        if type(image) is str:
            image = load_image(f"/scratch/noli1/kvasir_images/{sample['img_id']}.jpg")

        pope_prompt = sample["question"]
        ground_truth = sample["answer"]
        print(f"Processing sample {idx - from_index + 1}/{to_index - from_index + 1}")
        print(f"Question: {pope_prompt}")
        print(f"Ground truth: {ground_truth}")
        (
            answer,
            attention_tensor,
            visual_subgraphs,
            text_subgraphs,
            viusal_token_indices,
            text_token_indices,
            answer_token_indices,
            token_embeddings,
            sequence,
        ) = process_weights(model, processor, image_processor, context_len, image, pope_prompt)
        end = time.perf_counter()
        attention_file = f"{save_folder}/attention_{idx}.npz"
        visual_file = f"{save_folder}/visual_subgraphs_{idx}.npz"
        text_file = f"{save_folder}/text_subgraphs_{idx}.npz"
        embedding_file = f"{save_folder}/embedding_{idx}.npz"
        visual_arrays = {}

        for layer_idx, heads in visual_subgraphs.items():
            for head_idx, submat in enumerate(heads):
                visual_arrays[f"layer_{layer_idx}_head_{head_idx}"] = submat
        np.savez_compressed(visual_file, **visual_arrays)
        text_arrays = {}
        for layer_idx, heads in text_subgraphs.items():
            for head_idx, submat in enumerate(heads):
                text_arrays[f"layer_{layer_idx}_head_{head_idx}"] = submat
        np.savez_compressed(text_file, **text_arrays)
        # breakpoint()
        print(f"Elapsed time: {end - start:.6f} seconds")
        if answer is None:
            print("Skipping sample due to no valid answer generated.\n")
            continue
        # ()
        print(f"Model answer: {answer}")

        np.savez_compressed(attention_file, attention_tensor)
        np.savez_compressed(embedding_file, token_embeddings)
        results.append(
            {
                "question": pope_prompt,
                "ground_truth": ground_truth,
                "prediction": answer,
                "is_hallucinating": False,
                "visual_subgraphs": visual_file,
                "text_subgraphs": text_file,
                "attention_file": attention_file,
                "embedding_file": embedding_file,
                "visual_indices": viusal_token_indices,
                "text_indices": text_token_indices,
                "answer_indices": answer_token_indices,
                "sequence": sequence,
            }
        )

    save_results(results, output_dir=save_folder)
    print(f"Final Results length: {len(results)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="huggingface daaset id")
    parser.add_argument("--split", help="split for the dataset")
    parser.add_argument("--from-index", help="from index", type=int, default=0)
    parser.add_argument("--to-index", help="to index: if empty than all", type=int)
    parser.add_argument("--n-samples", help="total number of samples", type=int, default=int(1e12))
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "llava-med-v1.5-mistral-7b",
            "medgemma-4b-it",
        ],
        default="llava-med-v1.5-mistral-7b",
    )

    args = parser.parse_args()
    dataset, split = args.dataset, args.split
    from_index, to_index = args.from_index, args.to_index
    n_samples = args.n_samples
    model_name = args.model

    if to_index is None:
        # if end is not given then, end is the n_samples from the start
        to_index = from_index + n_samples

    if from_index > to_index:
        raise Exception(f"From index cannot be greater than to index, got {from_index}>{to_index}")
    elif (to_index - from_index) > n_samples:
        raise Exception(
            f"Not enough samples between from and to index, got {n_samples} to take from index {from_index} to {to_index}"
        )

    print(f"Running model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Split: {split}")
    print(f"Start Index: {from_index}")
    print(f"End Index: {to_index}")

    model, processor, image_processor, context_len = load_models(model_name)

    process_dataset(
        model_name,
        model,
        processor,
        image_processor,
        context_len,
        dataset=dataset,
        split=split,
        from_index=from_index,
        to_index=to_index,
    )
