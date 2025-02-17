import pandas as pd  
import matplotlib.pyplot as plt

# Create a simple dataset
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 40],
    "City": ["New York", "Los Angeles", "Chicago", "Houston"]
}

# Convert it into a DataFrame
df = pd.DataFrame(data)

# Add a new column 'Salary'
df["Salary"] = [50000, 60000, 70000, 80000]

# Plotting Age vs Name
plt.figure(figsize=(8,6))
plt.bar(df["Name"], df["Age"], color='blue')
plt.xlabel('Name')
plt.ylabel('Age')
plt.title('Age of Each Person')
plt.show()