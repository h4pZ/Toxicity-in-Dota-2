import os
import json
import joblib
import numpy as np
import pandas as pd
from utils import get_project_path

# Parameters.
abs_project_path = get_project_path(__file__)
models_path = os.path.join(abs_project_path, "models")
data_path = os.path.join(abs_project_path, "data")
stages = ["early", "mid", "late"]

# Loading the data.
print("Loading the data")
dataframes_names = [
    f"dota2_chat_messages_en_{stage}.csv" for stage in stages]
data = [pd.read_csv(os.path.join(data_path, dataframes_names[i]),
                    usecols=["text"]) for i in range(len(dataframes_names))]

# Loading lda models.
models = list()

for stage in stages:
    print(f"Loading {stage} model")
    path = os.path.join(models_path, f"model_{stage}.joblib")
    model = joblib.load(path)
    models.append(model)

# Loading the toxicity scores.
print("Loading the topic toxic scores")
toxicity_scores_path = os.path.join(models_path, "toxicity_scores.joblib")
toxicity_scores = joblib.load(toxicity_scores_path)

# Estimating the topic distribution per stage.
print("Estimating the stage topic distribution")
stage_topic_dist = {}

for i, df in enumerate(data):
    stage = stages[i]
    X = df["text"].dropna().tolist()
    model = models[i]
    doc_topic_dist = model.transform(X)
    topic_dist = doc_topic_dist.mean(axis=0)
    stage_topic_dist[stage] = topic_dist


# Calculating the toxicity stage score.
print("Calculating the toxicity stage score")
toxicity_stage_score = {}

for stage in stages:
    topic_toxic_scores = np.array(toxicity_scores[stage])
    stage_toxicity = np.sum(topic_toxic_scores * stage_topic_dist[stage]) \
        / np.sum(stage_topic_dist[stage])
    toxicity_stage_score[stage] = stage_toxicity

print(toxicity_stage_score)

print("Saving the toxicity stage score as a json")
json_path = os.path.join(models_path, "toxicity_stage_score.json")

with open(json_path, "w") as f:
    json.dump(toxicity_stage_score, f, indent=4)

