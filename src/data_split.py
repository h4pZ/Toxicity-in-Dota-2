import os
import numpy as np
import pandas as pd
from utils import get_project_path


# Parameters.
abs_project_path = get_project_path(__file__)
data_path = os.path.join(abs_project_path, "data")

# Loading the data and slicing.
df = pd.read_csv(os.path.join(data_path, "dota2_chat_messages_en.csv"))
percentiles = [33, 66, 100]
stages = ["early", "mid", "late"]
time_cutoffs = np.percentile(df["time"], q=[33, 66, 100])

for i, percentile in enumerate(percentiles):
    df_name = f"dota2_chat_messages_en_{stages[i]}.csv"
    cutoff = time_cutoffs[i]

    if i == 0:
        df_temp = df.loc[df["time"] <= cutoff, :]
    else:
        prev_cutoff = time_cutoffs[i - 1]
        df_temp = df.loc[(df["time"] > prev_cutoff)
                         & (df["time"] <= cutoff), :]

    print(f"Saving dota 2 stage {stages[i]} file")
    df_temp.to_csv(os.path.join(data_path, df_name), index=False)
