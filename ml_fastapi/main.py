# not used explicitly; but if not improted causes issues with __mp_main__
import json
import multiprocessing
import os
import pickle
from typing import Dict, List, Optional, Union

import __mp_main__
import numpy as np
import pandas as pd
from catboost import Pool
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware

# from utils import Tokenize

# uvicorn использует multiprocessing
# pickle ругается, если в Tokenize не в __mp_main__.Tokenize
# https://stackoverflow.com/questions/62953477
# fastapi-could-not-find-model-defintion-when-run-with-uvicorn
# __mp_main__.Tokenize = Tokenize

logger.add(
    "./logs/log.log", rotation="50 MB", retention=5,
)


class Text(BaseModel):
    text: str = ""

HF_TOKEN = os.environ.get("HF_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

columns = ["Исполнитель", "Группа тем", "Тема"]

models = dict()
for col in columns:
    filename = f"catboost_{col}.pcl"
    if not os.path.exists(filename):
        continue
    with open(filename, "rb") as f:
        models[col] = pickle.load(f)


def predict_catboost(input_text):
    with open("group_to_themes.json") as f:
        group_to_themes = json.load(f)
    outputs = dict()
    prev_tag = None
    for col, model in models.items():
        probas = model.predict_proba(input_text)
        if col == "Тема":
           allowed_tags = set(group_to_themes[prev_tag])
           for ci, c in enumerate(model.classes_):
               if c not in allowed_tags:
                   probas[ci] = -1000
        max_tag_id = np.argmax(probas)
        tag = model.classes_[max_tag_id]
        prev_tag = tag
        proba = np.max(probas)
        outputs[col] = {"tag": tag, "proba": round(float(proba), 2)}
    return outputs


@app.post("/product_single")
def product_single(text: Text) -> Dict:
    """Эндпойнт FastApi для выдачи предсказаний по тексту

    Args:
        text (Text): текст документа

    Returns:
        Dict: словарик с предсказанием
    """
    input_text = [text.text]
    outputs = predict_catboost(input_text)
    return outputs
