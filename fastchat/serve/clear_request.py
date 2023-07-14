"""
Use FastChat with request.

Usage:
from fastchat.serve.clear_request import get_response

response = get_response("Hello! Who are you?")
print(response)
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args


@torch.inference_mode()
def get_response(prompt):
    model_path = "lmsys/vicuna-7b-v1.3"
    device = "cuda"
    num_gpus = 1
    max_gpu_memory = 1
    load_8bit = True
    cpu_offloading = False
    debug = False
    temperature = 0.7
    repetition_penalty = 1.0
    max_new_tokens = 512
    
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        debug=debug,
    )

    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    
    if "t5" in model_path and repetition_penalty == 1.0:
        repetition_penalty = 1.2
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    return outputs
