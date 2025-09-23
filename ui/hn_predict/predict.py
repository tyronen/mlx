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


class PredictionResponse(BaseModel):
    prediction: float


def make_prediction(input_data):
    prediction = predictor.predict(input_data)
    return PredictionResponse(prediction=prediction)


# Define the prediction endpoint
def predict_direct(request: HNPostData):
    return make_prediction(request)
