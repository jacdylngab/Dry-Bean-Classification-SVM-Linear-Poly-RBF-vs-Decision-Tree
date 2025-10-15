import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Results.csv")

# Melt the dataframe into long form for seaborn
df_melted = df.melt(id_vars="Model", 
                    value_vars=["Macro_F1 (Train)", "Macro_F1 (Test)"],
                    var_name="Results", 
                    value_name="F1_Score")

sns.barplot(data=df_melted, x="Model", y="F1_Score", hue="Results", palette=["gold", "green"])

plt.title("Train vs Test Macro F1 Scores")
plt.tight_layout()

plt.legend(loc='lower left', title='Results')

plt.savefig("F1_Macro_bar_chart.png")
plt.show()
