#!/bin/bash
echo "Preprocesing the data"
echo "Obtaining the languages of each text sent"
python ./src/preprocessing_jl.py

echo "Creating dataset with only english comments and cleaning it"
python ./src/get_en_data.py

echo "Generating datasets for early, mid and late game"
python ./src/data_split.py

echo "Generating some EDA plots"
python ./src/EDA.py

echo "Training LDA for early, mid and late game"
python ./src/train.py

echo "Calculating toxicity scores of the topics for early, mid and late game"
python ./src/eval.py

echo "Calculating the Toxicicty Stage Score for each model"
python .src/stage_toxicity.py
