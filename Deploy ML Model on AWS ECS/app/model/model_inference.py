# Importing packages

import re
import pickle
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

#----------------------------------------------------------------------#

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/lang_trained_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

#----------------------------------------------------------------------#

classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]


def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]


#----------------------------------------------------------------------#

if __name__ == '__main__':
    text = 'Ciao, come stai?'
    detect = predict_pipeline(text)
    print('Prediction :', detect)
