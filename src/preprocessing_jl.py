import os
from joblib import Parallel, delayed
from langdetect import detect
import pandas as pd
from utils import get_project_path


# Parameters.
abs_project_path = get_project_path(__file__)
data_path = os.path.join(abs_project_path, "data")

# Loading the data.
df = pd.read_csv(os.path.join(data_path, "dota2_chat_messages.csv"))

# Replacing nan's on the text for empty strings.
df = df.fillna("")


def check_lang(text):
    """Will return the language corresponding to the
    input text"""
    try:
        lang = detect(text)
    except:
        lang = "nal"

    return lang


# IMPORTANT: haven't run this but discovered about joblib after doing the task
# with dask. Apparently is faster than dask since there is no graph overhead as in dask
# so from a few tests it seems that it might be faster than dask for about 2 hours.
languages = Parallel(n_jobs=8, verbose=11)(
    delayed(check_lang)(df["text"][i]) for i in range(df.shape[0]))
df["language"] = languages
df.to_csv("./data/dota2_chat_messages_lang_jl.csv", index=False)

