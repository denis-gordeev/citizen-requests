# not used explicitly; but if not improted causes issues with __mp_main__
import multiprocessing
import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catboost import Pool
from pydantic import BaseModel

# uvicorn использует multiprocessing
# pickle ругается, если в Tokenize не в __mp_main__.Tokenize
# https://stackoverflow.com/questions/62953477
# fastapi-could-not-find-model-defintion-when-run-with-uvicorn
# __mp_main__.Tokenize = Tokenize

class ClaimsSubject():
    columns = ["Исполнитель", "Группа тем", "Тема"]
    models = dict()

    def __init__(self):
        for col in self.columns:
            filename = f"catboost_{col}.pcl"
            if not os.path.exists(filename):
                continue
            with open(filename, "rb") as f:
                self.models[col] = pickle.load(f)

    def perform(self, text):
        outputs = dict()
        input_text = [text]
        for col, model in self.models.items():
            tag = model.predict(input_text)[0]
            proba = model.predict_proba(input_text)[0]
            outputs[col] = {"tag": tag, "proba": round(float(proba), 2)}
        return outputs