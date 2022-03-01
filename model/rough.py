import matplotlib.pyplot as plt

# Define data values
x = [7, 14, 21, 28, 35, 42, 49]
y = [5, 12, 19, 21, 31, 27, 35]
z = [3, 5, 11, 20, 15, 29, 31]

x = [1],
y = [2],
z = [3]

fig, axs = plt.subplots(2,2)
fig.suptitle('Vertically stacked subplots')

# Plot a simple line chart
axs[0][0].plot(x,y)
# Plot another line on the same chart/graph
axs[0][1].plot(x,z)

axs[1][0].plot(x,y)
axs[1][0].set_ylabel("tp")

plt.show()