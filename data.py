import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')

print(data)
plt.scatter(data.x, data.y)
plt.show()