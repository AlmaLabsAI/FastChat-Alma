import json
# cerebrium deploy lama7 --hardware A10 --api-key private-25d03f41962068aed23a
from typing import List

from vllm import LLM, SamplingParams
from pydantic import BaseModel

from fastchat.serve.huggingface_api import generate_response


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="lmsys/vicuna-7b-v1.3", tokenizer="hf-internal-testing/llama-tokenizer")


class Message(BaseModel):
    role: str
    content: str

class Item(BaseModel):
    prompt: List[Message]# Here's the modification, we're expecting a list of dictionaries

def format_prompt(messages):
    formatted_prompt = ""
    for message in messages:
        formatted_prompt += f"{message.role}: {message.content}\n"
    return formatted_prompt


def predict(item, run_id, logger):
    item = Item(**item)

    response = generate_response(item.prompt)
    print(response)

    return {"Prediction": response}
