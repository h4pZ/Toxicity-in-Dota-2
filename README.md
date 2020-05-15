# Measuring toxicity in Dota 2

# Introduction
Dota 2 is toxic af

# Purpose
Hypothesis: early and late game in dota 2 are more toxic than mid game

To run the project please download and extract the data set https://www.kaggle.com/romovpa/gosuai-dota-2-game-chats
in the folder data (if it doesn't exist create it `mkdir data`).

```bash
mkdir models
sh run.sh
```

# TODO

# DONE
- Add script to get only text in english.
- Get some statistics about the dataset.
- Define a time for early, mid and late game: Since the early, mid and late game depends on the pace on each game and all of them differ, the early, mid and late game times are define using the tertiles of the time distribution of the matches. It's assumed that on average this will reflect the appropiate times for each stage of the game.
- Research about toxicity measuring models on academia and kaggle competitions.
- Look for toxicity models from kaggle. Using those, decide whether to clean all data from non english commentaries. The answer is yes, I'm gonna remove all non english messages.
- Remove languages that aren't handled by the toxicity model.
- Research about topic modelling and use the best model for this case.
- Create and train 3 models for the early, mid and late game.
- Use the generated topics on the early, mid and late game and measure it's toxicity given the toxicity model above.
- Explore the topics and the toxicity scores to add to the analysis.
- Write the report
- Upload project to github
