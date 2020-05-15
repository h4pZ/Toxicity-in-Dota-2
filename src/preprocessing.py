import os
from langdetect import detect
import pandas as pd
import dask.dataframe as dd
from utils import get_project_path

# Parameters.
abs_project_path = get_project_path(__file__)
df_path = os.path.join(abs_project_path, "/data/dota2_chat_messages.csv")

# Loading the data.
df = pd.read_csv(df_path)

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


# Creating the dask dataframe and getting the languages
# for each text.
# NOTE: This takes around 12 hours on a i7 7700k.
ddf = dd.from_pandas(df, npartitions=8000)
languages = ddf["text"].apply(check_lang, meta=(None, "object"))
df["language"] = languages
df.to_csv("./data/dota2_chat_messages_lang.csv", index=False)
