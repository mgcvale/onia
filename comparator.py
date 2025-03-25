import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv("predicoes0.872.csv")
df2_name = "predictions_final.csv"
df2 = pd.read_csv(df2_name)

total = len(df1)
eq = (df1['target'] == df2['target']).sum()
print(eq /total)

dist1 = df1["target"].value_counts(normalize=True).sort_index()
dist2 = df2["target"].value_counts(normalize=True).sort_index()

dist_df = pd.DataFrame({"Classe": dist1.index, "872": dist1, df2_name: dist2}).reset_index(drop=True)
dist_df = dist_df.melt(id_vars="Classe", var_name="Modelo", value_name="Proporcao")

plt.figure(figsize=(8, 5))
sns.barplot(data=dist_df, x="Classe", y="Proporcao", hue="Modelo", palette="viridis")
plt.xlabel("Classe")
plt.ylabel("Proporcao")
plt.title("Distribuicao de classes")
plt.show()

conf_matrix = pd.crosstab(df1["target"], df2["target"])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")

plt.xlabel("predicoes do 872")
plt.ylabel("df2_name")
plt.title("Confusion Matrix")
plt.show()