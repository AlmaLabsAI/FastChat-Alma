import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args

def generate_response(message_list, sampling_params, llm, model_path='lmsys/vicuna-13b-v1.3', temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, debug=False):

    @torch.inference_mode()
    def _main(message_list, sampling_params, llm, model_path, temperature, repetition_penalty, max_new_tokens, debug):
        conv = get_conversation_template("lmsys/vicuna-7b-v1.3")

        for message in message_list:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        prompt = conv.get_prompt()
        additional_text = "</s>ASSISTANT:"
        prompt = prompt + additional_text

        print(f"PROMPT = {prompt}")
        generated_text = "TEST"


        outputs = llm.generate(prompt, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text

        generated_text = generated_text.replace("</s>", "")
        return generated_text

    return _main(message_list, sampling_params, llm, model_path, temperature, repetition_penalty, max_new_tokens, debug)


def initialize_llm(model='lmsys/vicuna-7b-v1.3', tokenizer='hf-internal-testing/llama-tokenizer'):
    llm = LLM(model=model, tokenizer=tokenizer)
    return llm
