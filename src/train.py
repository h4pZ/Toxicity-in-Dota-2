import os
import pandas as pd
import joblib
from utils import get_project_path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Parameters.
seed = 4444
verbose = True
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

# Grid search over each of the data sets.
for i, stage in enumerate(stages):
    print(f"TRAINING {stage}")

    # Setting up the data for each stage.
    X = data[i]["text"].tolist()

    # Creating the pipeline for the gridsearch.
    steps = [("vect", CountVectorizer()),
             ("lda", LatentDirichletAllocation(random_state=seed,
                                               max_iter=15))]
    pipe = Pipeline(steps=steps, verbose=verbose)
    params = {"vect__ngram_range": [(1, 1), (1, 2), (1, 3)],
              "lda__n_components": [3, 6, 9]}

    # Grid search for cv-k = 5.
    gs = GridSearchCV(estimator=pipe,
                      param_grid=params,
                      verbose=verbose,
                      n_jobs=-1)
    gs.fit(X)
    n_topics = gs.best_estimator_.named_steps["lda"].components_.shape[0]

    # Saving the best model.
    # GridSearchCV by default saves the best model (refit=True)
    # which is trained on all the data set.
    print(f"SAVING {stage}")
    print(f"Number of topics: {n_topics}")
    joblib.dump(gs.best_estimator_,
                os.path.join(models_path, f"model_{stage}.joblib"))
