from pydantic import BaseModel
from transformers import pipeline
from typing import Optional  # Добавьте эту строку

from fastchat.serve.clear_request import get_response

logger.info("am here")
output = get_response("TEST")


class Item(BaseModel):
    prompt: str
    max_length: Optional[int] = 100


def predict(item, run_id, logger):
    item = Item(**item)
    output = get_response(item.prompt)
    logger.info("Generated text: " + output)
    return output


