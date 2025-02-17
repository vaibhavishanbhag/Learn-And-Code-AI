import pandas as pd  

# Create a simple DataFrame
data = {"Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]}
df = pd.DataFrame(data)

# Add a new column 'Salary'
df["Salary"] = [50000, 60000, 70000]

# Filter rows where Age is greater than 25
df_filtered = df[df["Age"] > 25]

# Display summary statistics for numerical columns
print(df.describe())

# Display detailed information about the DataFrame
print(df.info())
