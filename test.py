import matplotlib.pyplot as plt
import numpy as np

# Create some data for the plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot the data on the first subplot
ax1.plot(x, y1)

# Plot the data on the second subplot
ax2.plot(x, y2)

# Show the plot
plt.show()
