import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

df = pd.read_parquet("sampled_data.parquet")
print(df.columns)

results = pd.read_csv("train.csv")
print(results.columns)
results.dropna(subset=["sii"], inplace=True)

average_light = df[["id", "light"]].groupby(["id"]).mean()
print(average_light)
# print(average_light["id"])
print(results["id"])
avg_light_per_id = average_light.join(results[["id", "sii"]], on="id", how="inner")
print(avg_light_per_id)

fig, ax = plt.subplots()
ax.scatter(avg_light_per_id["light"], avg_light_per_id["sii"])
plt.show()
