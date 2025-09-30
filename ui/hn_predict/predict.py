import os

from pydantic import BaseModel

from common import utils
from .model_handler import get_predictor

MODEL_TYPE = os.getenv("MODEL_TYPE", "double")

utils.setup_logging()

# Initialize the model -- change as desired
predictor = get_predictor()


# Define the request and response pydantic models
class HNPostData(BaseModel):
    by: str
    title: str
    url: str
    time: int
    score: int | None = None
    # add more if helpful


def predict_direct(request: HNPostData):
    return predictor.predict(request.model_dump())
