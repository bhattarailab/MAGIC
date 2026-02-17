import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]="/scratch/noli1"
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

#from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image

import torch
from PIL import Image
import warnings # Added to suppress a warning from PIL
torch.cuda.empty_cache()
from datasets import load_dataset
from utils import save_results
import numpy as np
from collections import defaultdict
#"LLAVA-1.5-7b-hf"#"#
#from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from io import BytesIO
import base64
import random
from transformers import set_seed as hf_set_seed
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="huggingface daaset id")
parser.add_argument ("--split", help="split for the dataset")
parser.add_argument("--from_index", help="from index", type=int, default=0)
parser.add_argument("--to_index", help="to index: if empty than all", type=int)
parser.add_argument("--n-samples", help="total number of samples", type=int, default=int(1e12))
parser.add_argument("--model", type=str,
	choices=[
				"LLAVA-1.5-7b-hf",
				"Qwen2.5-VL-3B-Instruct",
				"llava-med-v1.5-mistral-7b",
				"CheXagent",
				"medgemma-4b-it",
	],
	default="llava-med-v1.5-mistral-7b"
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
elif (to_index-from_index) > n_samples:
		raise Exception(f"Not enough samples between from and to index, got {n_samples} to take from index {from_index} to {to_index}")

def load_image(path: str):
    with Image.open(path) as img:
        return img.convert("RGB")
    

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

# Call at the beginning
clear_cuda()

def all_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)

all_seed(42)

def image_to_base64(image_path_or_pil):
    """
    Converts a PIL Image or image path to a base64 string suitable for VLM input.
    """
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"

def load_models(model_name):    
    if model_name == "LLAVA-1.5-7b-hf": 
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto" 
        )
        processor = AutoProcessor.from_pretrained(model_id)
    if model_name=="Qwen2.5-VL-3B-Instruct":
        model_id="Qwen/Qwen2.5-VL-3B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            attn_implementation="eager",
            device_map="auto",
            max_memory={
        0: "40GiB",   # GPU 0
        1: "40GiB",   # GPU 1
    },
            trust_remote_code=True 
        )
        processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)
        return model, processor,None,None
    if model_name=="llava-med-v1.5-mistral-7b":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.mm_utils import (
						KeywordsStoppingCriteria,
						get_model_name_from_path,
						process_images,
						tokenizer_image_token,
				)
        from llava.model.builder import load_pretrained_model

        #breakpoint()
        model_name = get_model_name_from_path("microsoft/llava-med-v1.5-mistral-7b")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
                                                    model_path="microsoft/llava-med-v1.5-mistral-7b", 
                                                    model_base=None, 
                                                    model_name=model_name, 
                                                    load_4bit=False) #True )
        #breakpoint()
        return model, tokenizer,image_processor,context_len
    if model_name=="CheXagent":
        tokenizer = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
        
        model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True,torch_dtype=torch.float16, 
            device_map="auto" )  
        print(generation_config)
        return model, tokenizer,None,None

    if model_name=="medgemma-4b-it":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from huggingface_hub import login
        login(token="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")

        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
    				torch_dtype=torch.bfloat16,
    				device_map="auto",
    				attn_implementation='eager'
				)
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        return  model,processor, None, None

def process_image(image,image_processor):
    from llava.mm_utils import process_images

    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)

def create_prompt(prompt: str):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import SeparatorStyle, conv_templates

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
    placeholder_mask = (input_ids == image_token_id)
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
        'visual_token_indices': visual_token_indices,
        'text_token_indices': text_token_indices,
        'answer_token_indices': answer_token_indices,
        'tokens_per_image': tokens_per_image,
    }
def process_weights(model, processor, image_processor,context_len,image, question):
    if  model_name=="llava-med-v1.5-mistral-7b":
        from llava.mm_utils import (
						KeywordsStoppingCriteria,
						tokenizer_image_token,
				)
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import SeparatorStyle, conv_templates

        image_tensor = process_image(image,image_processor)
        prompt, conv = create_prompt(question)
        input_ids = (
        tokenizer_image_token(prompt, processor, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2breakpoint()
        max_new_tokens = 256
        image_token_id=-200
        placeholder_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[1]
        generated_ids = input_ids.clone()

        stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=processor, input_ids=input_ids
    )

        with torch.no_grad():
            generated_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            #pad_token_id=processor.eos_token_id,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
            prompt_len = input_ids.shape[-1]
            sequence = torch.cat([input_ids, generated_ids], dim=1)
            output = model(
            sequence,
            images=image_tensor.unsqueeze(0).half().cuda(),
            output_attentions=True,
            use_cache=False
        )
        attention=output.attentions
        attention_seq_len = output.attentions[0].shape[-1]
        indices = get_token_type_indices(
    input_ids=input_ids,
    generated_ids=sequence,
    attention_seq_len=attention_seq_len
)
        #breakpoint()
        answer_token_indices = indices['answer_token_indices']
        text_token_indices = indices['text_token_indices']
        visual_token_indices = indices['visual_token_indices']        
        #breakpoint()
        layers = len(output.attentions)
        num_heads = output.attentions[0].shape[1]
        visual_subgraphs = {}
        text_subgraphs = {}
        #breakpoint()
        for layer_idx in range(layers):
            layer_attn = output.attentions[layer_idx][0].detach().cpu().numpy()  # shape (num_heads, seq_len, seq_len)
            visual_subgraphs[layer_idx] = []
            text_subgraphs[layer_idx] = []

            for head_idx in range(num_heads):
                attn = layer_attn[head_idx]
                visual_submat = attn[np.ix_(answer_token_indices, visual_token_indices)]
                text_submat = attn[np.ix_(answer_token_indices, text_token_indices)]
                visual_subgraphs[layer_idx].append(visual_submat)
                text_subgraphs[layer_idx].append(text_submat)
        attention = [attn[0].cpu().numpy() for attn in output.attentions]
        #breakpoint()
        prompt_len = input_ids.shape[-1]
        # answer_ids = generated_ids[0][prompt_len:]
        answer_ids=generated_ids[0]
        answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
        sequence=sequence.detach().cpu().numpy()
        #breakpoint()
        return answer, attention,visual_subgraphs,text_subgraphs,visual_token_indices,text_token_indices,answer_token_indices,sequence
    if model_name=="LLAVA-1.5-7b-hf" :
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': f"Answer in yes or no. {question}"}
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        max_new_tokens = 100
        generated_ids = inputs["input_ids"].clone()
        initial_len = generated_ids.shape[1]
        image_token_str = "<image>" 
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token_str)
        visual_token_indices = (generated_ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                output = model(
                    input_ids=generated_ids,
                    pixel_values=inputs.get("pixel_values"),
                    output_attentions=True,
                    use_cache=True
                )
                next_token_logits = output.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if next_token_id.item() == processor.tokenizer.eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        answer_token_indices = list(range(initial_len, generated_ids.shape[1]))
        text_token_indices = [i for i in range(generated_ids.shape[1])
                              if i not in visual_token_indices and i not in answer_token_indices]
        layers = len(output.attentions)
        num_heads = output.attentions[0].shape[1]
        visual_subgraphs = {}
        text_subgraphs = {}

        for layer_idx in range(layers):
            layer_attn = output.attentions[layer_idx][0].detach().cpu().numpy()  # shape (num_heads, seq_len, seq_len)
            visual_subgraphs[layer_idx] = []
            text_subgraphs[layer_idx] = []

            for head_idx in range(num_heads):
                attn = layer_attn[head_idx]
                visual_submat = attn[np.ix_(answer_token_indices, visual_token_indices)]
                text_submat = attn[np.ix_(answer_token_indices, text_token_indices)]
                visual_subgraphs[layer_idx].append(visual_submat)
                text_subgraphs[layer_idx].append(text_submat)

        attention = [attn.cpu().numpy() for attn in output.attentions]
        prompt_len = inputs["input_ids"].shape[-1]
        answer_ids = generated_ids[0][prompt_len:]
        #()
        answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
        #
        return answer, attention,visual_subgraphs,text_subgraphs
    if model_name=="Qwen2.5-VL-3B-Instruct":
        image=image_to_base64(image)
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Answer as concisely as possible.:{question}"},
                    {'type': 'image',"image":image},
                ],
            },
        ]
        #image = Image.open(image).convert("RGB")
        prompt = processor.apply_chat_template(conversation, tokenize=False,add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
    text=[prompt],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")
        max_new_tokens = 75
        all_attentions = []
        generated_ids = inputs["input_ids"].clone()
        initial_len = generated_ids.shape[1]
        image_token_str = "<|image_pad|>"
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token_str)
        visual_token_indices = (generated_ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                output = model(
                    input_ids=generated_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    output_attentions=True, 
                    use_cache=True 
                )
                #
                logits = output.logits  # Only works if model has LM head
                #breakpoint()
                
                next_token_logits = output.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if next_token_id.item() == processor.tokenizer.eos_token_id or step==max_new_tokens-1:
                    break
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        #special_token_ids = set(processor.tokenizer.all_special_ids)
        #breakpoint()
        answer_token_indices = [i for i in range(initial_len, generated_ids.shape[1])]
        text_token_indices = [i for i in range(generated_ids.shape[1])
                            if i not in visual_token_indices
                            and i not in answer_token_indices
                            ]
        
        if len(answer_token_indices)==0:
            return None, None,None,None    
        layers = len(output.attentions)
        #breakpoint()
        num_heads = output.attentions[0].shape[1]
        #breakpoint()
        visual_subgraphs = {}
        text_subgraphs = {}
        #breakpoint()
        attention = [attn.cpu().numpy() for attn in output.attentions]
        prompt_len = inputs["input_ids"].shape[-1]
        answer_ids = generated_ids[0][prompt_len:]
        for layer_idx in range(layers):
            #print(layer_idx)
            layer_attn = output.attentions[layer_idx][0].detach().cpu().numpy()  # shape (num_heads, seq_len, seq_len)
            visual_subgraphs[layer_idx] = []
            text_subgraphs[layer_idx] = []

            for head_idx in range(num_heads):
                
                attn = layer_attn[head_idx]
                
                visual_submat = attn[np.ix_(answer_token_indices, visual_token_indices)]
                text_submat = attn[np.ix_(answer_token_indices, text_token_indices)]

                visual_subgraphs[layer_idx].append(visual_submat)
                text_subgraphs[layer_idx].append(text_submat)
        answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
        sequence=sequence.detach().cpu().numpy()
        #breakpoint()
        return answer, attention,visual_subgraphs,text_subgraphs

    if model_name=="medgemma-4b-it":
        prompt = "Answer the question as concisely as possible: " + question
        messages = [
{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]
    }
        ]
        #breakpoint()
        inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=False,
        use_cache=True
    )
        #breakpoint()
        answer_ids = generated_ids[0][prompt_len:]
        full_sequence = generated_ids 
        with torch.no_grad():
            outputs = model(
            input_ids=full_sequence,
            pixel_values=inputs.get("pixel_values"),
            output_attentions=True,
            use_cache=False
        )
    
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        input_ids_list = full_sequence[0].cpu().tolist()
        visual_token_indices = [i for i, tid in enumerate(input_ids_list) if tid == image_token_id]
        text_token_indices = [i for i in range(prompt_len) if i not in visual_token_indices]
        answer_token_indices = list(range(prompt_len, len(input_ids_list)))
        #breakpoint()
        layers = len(outputs.attentions)
        num_heads = outputs.attentions[0].shape[1]
        visual_subgraphs = {}
        text_subgraphs = {}
        #breakpoint()
        layer_types = model.config.text_config.layer_types 
        global_layer_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        print(f"Extracted Global Layers at: {global_layer_indices}")
        global_attentions=[]
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
        sequence=generated_ids[0].detach().cpu().numpy()
        #breakpoint()
        return answer, global_attentions,visual_subgraphs,text_subgraphs,visual_token_indices,text_token_indices,answer_token_indices,sequence 

import re    
def normalize_yes_no(text: str):
    """Extract yes/no from a text response."""
    text = text.lower().strip()
    if re.match(r'^\s*yes\b', text):
        return "yes"
    elif re.match(r'^\s*no\b', text):
        return "no"
    if "normal" in text or "unremarkable" in text:
        return "yes"
    if "abnormal" in text or "opacity" in text or "infiltrate" in text:
        return "no"
    return text  

def process_dataset(model, processor, image_processor=None,context_len=None,dataset="flaviagiammarino/vqa-rad",split='train', from_index=0, to_index=1e12):
    """0
   
     Process samples from  dataset 
    Args:
        model: Vision-language model
        processor: Model processor  
        split: One of 'random', 'popular', 'adversarial'
        num_samples: Number of samples to process
    """
    import time
    save_folder=f"/scratch/noli1/graph_hallucination_detection/results/{model_name}_{split}_Feb11_{dataset.split('/')[-1]}_from_{from_index}_to_{to_index}"
    os.makedirs(save_folder,exist_ok=True)
    dataset = load_dataset(dataset, split=split)
    results = []
    count=0
    
    to_index = min(to_index, len(dataset))
    
    for idx in range(from_index, to_index):
        #print(idx)
        start = time.perf_counter()
        sample = dataset[idx]
        image = sample['image']
        if type(image) is str:
            image = load_image(f"/scratch/noli1/kvasir_images/{sample['img_id']}.jpg")

        pope_prompt =  sample['question']
        ground_truth=sample['answer']
        print(f"Processing sample {idx + 1}/{to_index-from_index}")
        print(f"Question: {pope_prompt}")
        print(f"Ground truth: {ground_truth}")
        # if ground_truth.lower().strip() not in ['yes','no']:
        #     print("Skipping sample due to non-binary ground truth.\n")
        #     continue
        answer, attention_tensor,visual_subgraphs,text_subgraphs,viusal_token_indices,text_token_indices,answer_token_indices,sequence = process_weights(
            model, 
            processor,
            image_processor,
            context_len, 
            image, 
            pope_prompt
        )
        end=time.perf_counter()
        attention_file=f"{save_folder}/attention_{idx}.npz"
        visual_file = f"{save_folder}/visual_subgraphs_{idx}.npz"
        text_file = f"{save_folder}/text_subgraphs_{idx}.npz"
        visual_arrays = {}
        for layer_idx, heads in visual_subgraphs.items():
            for head_idx, submat in enumerate(heads):
                visual_arrays[f'layer_{layer_idx}_head_{head_idx}'] = submat
        np.savez_compressed(visual_file, **visual_arrays)
        text_arrays = {}
        for layer_idx, heads in text_subgraphs.items():
            for head_idx, submat in enumerate(heads):
                text_arrays[f'layer_{layer_idx}_head_{head_idx}'] = submat
        np.savez_compressed(text_file, **text_arrays)
        #breakpoint()
        print(f"Elapsed time: {end - start:.6f} seconds")
        if answer is None:
            print("Skipping sample due to no valid answer generated.\n")
            continue
        #()    
        print(f"Model answer: {answer}")
        #is_correct = normalize_yes_no(answer) == ground_truth.lower().strip()
        #print(f"Correct: {is_correct}\n")        
        np.savez_compressed(attention_file, attention_tensor)
        results.append({
            'question': pope_prompt,
            'ground_truth': ground_truth,
            'prediction': answer,
            'is_hallucinating': False,
            "visual_subgraphs": visual_file,
            "text_subgraphs": text_file,
            "attention_file": attention_file,
            "visual_indices":viusal_token_indices,
            "test_indices":text_token_indices,
            "answer_indices":answer_token_indices,
            "sequence": sequence
        })
        
    save_results(results,output_dir=save_folder)
    print(f"Final Results length:{len(results)}")
    accuracy = sum(r['is_hallucinating'] for r in results) / len(results)
    print(f"Accuracy: {accuracy:.2%}")    
    return results

image_path = "/work/FAC/HEC/DESI/yshresth/aim/sacharya/hallucination_aware_fine_tunning/Kavasir/kvasir-dataset-v2/dyed-lifted-polyps/0a7bdce4-ac0d-44ef-93ee-92dfc8fe0b81.jpg"
question = "What is the image about?"
if __name__ == "__main__":
    save_result = True
    load_result = False
    if save_result:
        print(f"Processing {n_samples} between {from_index} and {to_index} for {dataset} dataset on {split} split")

        model, processor,image_processor,context_len = load_models(model_name)
        process_dataset(model, processor,image_processor,context_len, split=split, from_index=from_index, to_index=to_index, dataset=dataset)
    if load_result:
        from utils import load_results
        print(split)
        info_text,info_image = load_results(f"/work/FAC/HEC/DESI/yshresth/aim/noli1/pickles/dec15_test.pkl")
        # print("Statistical Significance Plotting...")
        # from graph_utils import statistical_significance_plot
        # statistical_significance_plot(info_image, title="Visual AttnScore",save_path=f"{model_name}_visual_significance_heatmap_train.png",cache_path=f"{model_name}_visual_features_cache_{split}_normalized.pkl")
        # statistical_significance_plot(info_text, title="Text AttnScore",save_path=f"{model_name}_text_significance_heatmap_train.png",cache_path=f"{model_name}_text_features_cache_{split}_normalized.pkl")
