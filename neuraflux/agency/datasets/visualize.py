import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv("dynamic_pricing_2023.csv", index_col=0, parse_dates=True)
plt.plot(df.price)
plt.show()