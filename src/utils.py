import os
import time
import numpy as np
import json
import requests


def get_project_path(module_path):
    """Gets the absolute path of the project folder.

    Parameters
    ----------
    module_path : str
        __file__ variable of the file that calls this function.

    Returns
    -------
    abs_project_path : str
        Absolute path of the project
    """
    src_path = os.path.dirname(module_path)
    abs_src_path = os.path.abspath(src_path)
    abs_project_path = os.path.dirname(abs_src_path)

    return abs_project_path


def get_top_words(model, feature_names, n_top_words):
    """Get the top words on each topic from a model.
    Taken from and modified a little bit:
    https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    """
    topics = list()
    weights = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    for topic_idx, topic in enumerate(model.components_):
        words = [feature_names[i]
                 for i in topic.argsort()[:-n_top_words - 1:-1]]
        words_weights = [weights[topic_idx, j]
                         for j in weights[topic_idx, :].argsort()[:-n_top_words - 1:-1]]
        words_dict = {key: value for key, value in zip(words, words_weights)}
        topics.append(words_dict)

    return topics


def get_toxicity(text, api_key):
    """Gets the toxicity for some text from https://www.perspectiveapi.com

    Parameters
    ----------
    text : str
        Text to evaluate the toxicity
    api_key : str
        API key for perspective

    Returns
    -------
    probability : float
        The probability of the text being toxic.
    """
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    data_dict = {"comment": {"text": text},
                 "languages": ["en"],
                 "requestedAttributes": {"TOXICITY": {}}}
    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content)
    probability = response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

    # Put to sleep so that Google doesn't bother me
    # with the 1 QPS (queries per second)
    time.sleep(1.3)

    return probability


def get_score(topic_top_words, api_key):
    """Computes the weighted toxicity score for a topic.

    Parameters
    ----------
    topic_top_words : dict
        Dictionary containing as keys the topic words and as values
        the weights from the LDA.
    api_key : str
        API key for perspective API.

    Returns
    -------
    score : float
        Weighted toxicity score.
    """
    words = topic_top_words.keys()
    weights = np.array(list(topic_top_words.values()))
    toxic_scores = [get_toxicity(word, api_key) for word in words]
    toxic_scores = np.array(toxic_scores)
    score = np.sum(toxic_scores * weights / weights.sum())

    return score
