import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 50]

# Creating a line plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label="Sales Growth")

# Adding labels and title
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales Growth Over Days")
plt.legend()
plt.show()

# Generating random data
data = np.random.randn(1000)

# Creating a histogram
sns.histplot(data, bins=30, kde=True, color="purple")

# Showing the plot
plt.title("Data Distribution")
plt.show()


import pandas as pd

# Creating sample data
df = pd.DataFrame({
    "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Test Score": [50, 55, 60, 65, 70, 72, 78, 85, 88, 95]
})

# Creating scatter plot
sns.scatterplot(x=df["Study Hours"], y=df["Test Score"], color="red")

# Adding labels
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.title("Study Hours vs. Test Score")
plt.show()

