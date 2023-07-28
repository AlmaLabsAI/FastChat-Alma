from typing import List, Dict

from vllm import LLM, SamplingParams
from pydantic import BaseModel
from typing import Optional

from fastchat.serve.clear_request import generate_response, initialize_llm
from vllm.sampling_params import SamplingParams

llm = initialize_llm()
top_p = 1
max_new_tokens = 1024
temperature = 1
top_p = max(top_p, 1e-5)



sampling_params = SamplingParams(n=1,
    temperature=temperature,
    top_p=top_p,
    use_beam_search=False,
    max_tokens=max_new_tokens,
)


class Item(BaseModel):
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 10
    repetition_penalty: Optional[float] = 1.0


def predict(item, run_id, logger):
    print(f"item = {item}")
    item = Item(**item)
    response = generate_response(item.prompt, sampling_params, llm)
    print(response)
    return {"Prediction": "test_for_test"}
