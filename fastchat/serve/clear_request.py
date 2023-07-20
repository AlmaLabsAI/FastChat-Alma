# Example
#
# from fastchat.serve.huggingface_api import generate_response
#
# message = "Hello! Who are you?"
# response = generate_response(message)
# print(response)



import argparse
import json
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args

def generate_response(message, model_path='lmsys/vicuna-7b-v1.3', temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, debug=False):

    @torch.inference_mode()
    def _main(message, model_path, temperature, repetition_penalty, max_new_tokens, debug):

        sampling_params = SamplingParams(temperature=temperature, top_p=0.95)
        llm = LLM(model=model_path, tokenizer="hf-internal-testing/llama-tokenizer")

        msg = message

        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        outputs = llm.generate(prompt, sampling_params)
        generated_text = "TEST"
        for output in outputs:
            generated_text = output.outputs[0].text

        return generated_text

    return _main(message, model_path, temperature, repetition_penalty, max_new_tokens, debug)


