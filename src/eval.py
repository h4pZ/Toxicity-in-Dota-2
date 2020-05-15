import os
import json
import joblib
from utils import get_top_words, get_score, get_project_path


# Parameters.
with open("token.json", "r") as f:
    api_key = json.load(f)["token"]

abs_project_path = get_project_path(__file__)
models_path = os.path.join(abs_project_path, "models")
data_path = os.path.join(abs_project_path, "data")
stages = ["early", "mid", "late"]
n_top_words = 10

# Loading models.
models = list()

for stage in stages:
    print(f"Loading {stage} model")
    path = os.path.join(models_path, f"model_{stage}.joblib")
    model = joblib.load(path)
    models.append(model)

# Getting the top words on each topic for each model.
# NOTE: It is important to note that for all models the
# best number of topics was 3.
models_top_words = list()

for i, stage in enumerate(stages):
    print(f"Getting top words of {stage} game")

    # Getting the pipeline attributes.
    vectorizer = models[i].named_steps["vect"]
    lda_model = models[i].named_steps["lda"]
    feature_names = vectorizer.get_feature_names()

    # Getting and saving the top words for each model.
    top_words = get_top_words(lda_model, feature_names, n_top_words)
    models_top_words.append(top_words)

# Obtaining the toxicity score per model and per topic.
# The toxicity_scores is going to be a dictionary that as keys
# is going to have the stages of the game and as keys a list
# which is going to be populated with the toxicity score for each topic.
# So the first value on the list corresponds to the toxicity of the
# first topic.
toxicity_scores = {stage: list() for stage in stages}

for i, stage in enumerate(stages):
    top_words = models_top_words[i]
    lda_model = models[i].named_steps["lda"]

    for topic_idx in range(lda_model.components_.shape[0]):
        print(f"Toxicity scores for the {stage} game and topic {topic_idx}")
        topic_top_words = top_words[topic_idx]
        score = get_score(topic_top_words, api_key)

        # Add the score to toxicity_scores.
        toxicity_scores[stage] = toxicity_scores[stage] + [score]


# Saving the toxicity_scores and models_top_words.
print("Saving the models_top_words")
joblib.dump(models_top_words, os.path.join(
    models_path, "models_top_words.joblib"))
print("Saving the toxicity scores")
joblib.dump(toxicity_scores, os.path.join(
    models_path, "toxicity_scores.joblib"))

print("Model top words")
print(models_top_words)
print("Toxicity scores")
print(toxicity_scores)
