# not used explicitly; but if not improted causes issues with __mp_main__
import json
import multiprocessing
import os
import pickle
from typing import Dict, List, Optional, Union

import __mp_main__
import numpy as np
import pandas as pd
import torch
from catboost import Pool

from claim_ner import get_ner_format
from claim_letter import giga_letter

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from setfit import SetFitModel
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sigmoid = nn.Sigmoid()


with open("theme_group_label_encoder.pcl", "rb") as f:
    bert_label_encoder = pickle.load(f)

bert_classes = {c: ci for ci, c in enumerate(bert_label_encoder.classes_)}


def inference(model, message, **kwargs):
    input_ids, attention_mask = tokenize(message, **kwargs)
    preds = model(input_ids, attention_mask)
    logits = sigmoid(preds["logits"])[0].cpu().detach().numpy()
    return logits


def tokenize(message, device="cuda", **kwargs):
    text_enc = tokenizer.encode_plus(
        message,
        None,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors="np",
    )
    input_ids = torch.tensor(text_enc["input_ids"], dtype=torch.long).to(device)
    attention_mask = torch.tensor(text_enc["attention_mask"], dtype=torch.long).to(device)
    return input_ids, attention_mask


logger.add(
    "./logs/log.log",
    rotation="50 MB",
    retention=5,
)


class Text(BaseModel):
    text: str = ""

class Request(BaseModel):
    incident: str = ''
    subject: str = ''
    person: str = ''

HF_TOKEN = os.environ.get("HF_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", problem_type="single_label_classification")

group_roberta_model = AutoModelForSequenceClassification.from_pretrained(
    "denis-gordeev/xlm-roberta-base-theme-group", problem_type="single_label_classification", use_auth_token=HF_TOKEN
).to("cuda")

columns = ["Исполнитель", "Группа тем", "Тема"]

setfit_model_names = {
    "Исполнитель": "denis-gordeev/citizen-request-performer-labse",
    "Группа тем": "denis-gordeev/citizen-request-theme-group",
    "Тема": "denis-gordeev/citizen-request-theme-labse",
}

setfit_models = dict()
for model_name, hf_path in setfit_model_names.items():
    model = SetFitModel.from_pretrained(hf_path, use_auth_token=HF_TOKEN)
    with open(f"setfit_classes_{model_name}.json") as f:
        classes = json.load(f)
    setfit_models[model_name] = {"model": model, "classes": classes}

models = dict()
for col in columns:
    filename = f"catboost_{col}.pcl"
    if not os.path.exists(filename):
        continue
    with open(filename, "rb") as f:
        models[col] = pickle.load(f)


def predict_catboost(input_text: list[str], add_setfit=True):
    with open("group_to_themes.json") as f:
        group_to_themes = json.load(f)
    outputs = dict()
    prev_tag = None
    for col, model in models.items():
        probas = model.predict_proba(input_text)
        if col == "Группа тем":
            bert_probas = inference(group_roberta_model, input_text[0])
            bert_probas = [bert_probas[bert_classes[c]] for c in model.classes_]
            bert_probas = np.array(bert_probas)
            probas += bert_probas
            probas /= 2
        if add_setfit and col in setfit_models:
            setfit_model = setfit_models[col]["model"]
            setfit_classes = setfit_models[col]["classes"]
            setfit_preds = setfit_model.predict_proba(input_text).cpu().detach().numpy()[0]
            setfit_proba = np.array([setfit_preds[setfit_classes[cl]] for cl in model.classes_])
            # do not use catboost for Тема
            # standalone SetFit is better here
            if col == "Тема":
                probas = setfit_proba
            else:
                probas += setfit_proba
                probas /= 2
        # if col == "Тема":
        #    allowed_tags = set(group_to_themes[prev_tag])
        #    for ci, c in enumerate(model.classes_):
        #        if c not in allowed_tags:
        #            probas[ci] = -1000
        max_tag_id = np.argmax(probas)
        tag = model.classes_[max_tag_id]
        prev_tag = tag
        proba = np.max(probas)
        outputs[col] = {"tag": tag, "proba": round(float(proba), 2)}
    return outputs


text4 = "Директору ГБОУ «Школа № 1527» Кадыковой Елене Владимировне KadykovaEV@edu.mos.ru , 1527@edu.mos.ru Адрес: 115470, г.Москва, пр-кт Андропова, дом 17, корпус 5. Копия: в Южное окружное управление департамента образования города Москвы Адрес: г.Москва, Высокая улица, 14.        от Родителей учеников 6 «В» класса,) указанных ниже в подписях КОЛЛЕКТИВНАЯ ЖАЛОБА Уважаемая Елена Владимировна! С «19» октября 2020 года, с началом введения дистанционного обучения в системе МЭШ для 6-11 классов в связи с эпидемией COVID-19, наши дети, ученики 6 &quot;В&quot; класса не могут ежедневно подключиться к одному или нескольким урокам."


@app.post("/product_single")
def product_single(text: Text = Text(text=text4)) -> Dict:
    """Эндпойнт FastApi для выдачи предсказаний по тексту

    Args:
        text (Text): текст документа

    Returns:
        Dict: словарик с предсказанием
    """
    input_text = [text.text]
    outputs = predict_catboost(input_text)
    return outputs


@app.post("/product_ner")
def product_ner(text: Text = Text(text=text4)) -> Dict:
    """Эндпойнт FastApi для выдачи NER

    Args:
        text (Text): текст документа

    Returns:
        Html: разметка
    """
    input_text = text.text
    outputs = get_ner_format(input_text)
    return {"ner": outputs}


@app.post("/product_letter")
def product_letter(req: Request = Request()) -> Dict:
    outputs = giga_letter(req.incident, req.subject, req.person)
    return {"letter": outputs}
