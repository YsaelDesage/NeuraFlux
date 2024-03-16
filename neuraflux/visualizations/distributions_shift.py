import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; np.random.seed(2)
import random; random.seed(2)
import joypy

# Sample data
df = pd.DataFrame({'Time': np.random.normal(70, 100, 1200)+1200-np.arange(1200),
                   'var2': np.random.normal(250, 100, 1200),
                   'group': np.array(
                    ["t00"]*100+
                    ["t01"]*100+
                    ["t02"]*100+
                    ["t03"]*100+
                    ["t04"]*100+
                    ["t05"]*100+
                    ["t06"]*100+
                    ["t07"]*100+
                    ["t08"]*100+
                    ["t09"]*100+
                    ["t10"]*100+
                    ["t11"]*100
                    )})

fig, ax = joypy.joyplot(df, by = "group", column = "Time", fade = True, color="Gold")

plt.show()
