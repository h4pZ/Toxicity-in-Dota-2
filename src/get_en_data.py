import os
from tqdm import tqdm
import pandas as pd
import spacy
from utils import get_project_path


# Parameters and stuff.
tqdm.pandas()
abs_project_path = get_project_path(__file__)
data_path = os.path.join(abs_project_path, "data")

# Loading data.
print("Loading the data")
df = pd.read_csv(os.path.join(data_path, "dota2_chat_messages_lang.csv"))
df = (df.loc[df["language"] == "en", :]
        .reset_index(drop=True))

# Removing stop words and un used tokens.
nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    doc = nlp(text)
    tokens = list()

    for token in doc:
        if not token.is_stop and token.is_alpha and len(token) >= 3:
            tokens.append(token.text)

    clean_text = " ".join(tokens)

    return clean_text


# Cleaning the data.
# This takes about 3 hours. Tried working with joblib
# but spacy objects doesn't seem to work with it because
# they can't be pickled or something like that.
print("Removing stop words, non-alphabetic tokens and small tokens (len < 3)")
df["text"] = df["text"].progress_apply(preprocess)

# Saving the clean doc and removing obs that are empty.
df = (df.loc[df["text"] != "", :]
        .dropna()
        .reset_index(drop=True))
df.to_csv(os.path.join(data_path, "dota2_chat_messages_en.csv"), index=False)
