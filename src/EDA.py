import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_project_path


plt.style.use("hibm")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Parameters.
abs_project_path = get_project_path(__file__)
data_path = os.path.join(abs_project_path, "data")
df_path = os.path.join(data_path, "dota2_chat_messages_lang.csv")


# Loading the data.
df = pd.read_csv(df_path)

# Replacing nan's on the text for empty strings.
df = df.fillna("")

# Getting the distribution over time of the messages.
fig, ax = plt.subplots(figsize=(8, 5))

sns.distplot(df["time"], ax=ax)
ax.set(title="Message distribution over time",
       xlabel="Time (seconds)",
       ylabel="Density")

fig.savefig("./results/imgs/message_distribution_time.png")
plt.close()

# Getting the distribution of messages per player slot.
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(df["slot"], palette="viridis", ax=ax)
ax.set(title="Message distribution per player slot",
       xlabel="Player slot",
       ylabel="Message count")

fig.savefig("./results/imgs/message_distribution_per_player_slot.png")
plt.close()

# Getting some other statistics.
n_matches = df["match"].nunique()

df["count"] = 1
df_matches = df[["match", "count"]].groupby(by="match").sum()
avg_messages_match = np.mean(df_matches)[0]

fig, ax = plt.subplots(figsize=(8, 5))

sns.distplot(df_matches, ax=ax)
ax.text(x=1000, y=0.022,
        s=f"Average messages per match: {avg_messages_match:.2f}")
ax.set(title="Message distribution per match",
       xlabel="Messages per match",
       ylabel="Density")

fig.savefig("./results/imgs/message_distribution_per_match.png")
plt.close()

# Getting the distribution of the length of the messages.
df["text_length"] = df["text"].apply(lambda x: len(x))

fig, ax = plt.subplots(figsize=(8, 5))

sns.distplot(df["text_length"], ax=ax)
ax.set(title="Message length distribution",
       xlabel="Message length",
       ylabel="Density")

fig.savefig("./results/imgs/message_length_distribution.png")
plt.close()


# Getting the language distribution over the messages.
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(df["language"],
              palette="viridis",
              ax=ax,
              orient="v",
              order=df["language"].value_counts().index[:20])
ax.set(title="Top 20 languages used in the text chat",
       xlabel="Language",
       ylabel="Language count")

fig.savefig("./results/imgs/language_distribution.png")
plt.close()
