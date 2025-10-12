import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Results.csv")

plt.bar(df["Model"], df["Macro_F1 (Test)"])

plt.xlabel('Model')
plt.ylabel('F1 Macro')
plt.title('F1 Macro bar chart')
plt.savefig('F1 Macro bar chart')
plt.show()
