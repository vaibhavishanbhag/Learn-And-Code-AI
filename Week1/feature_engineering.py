# Handle missing values
import pandas as pd
import numpy as np

# Sample dataset
data = {'Age': [25, np.nan, 30, np.nan, 40], 'Salary': [50000, 60000, np.nan, 80000, 90000]}
df = pd.DataFrame(data)

# Filling missing values with mean
df.fillna(df.mean(), inplace=True)

print(df)


# Creating New Features

df['Date'] = pd.to_datetime(['2024-01-01', '2024-02-10', '2024-03-15', '2024-04-20', '2024-05-25'])
df['Month'] = df['Date'].dt.month  # Extracting the month
print(df)


# Encoding Categorical Data

from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red']})

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['Color']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
print(encoded_df)


# Scaling Numerical Data

from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({'Height': [160, 170, 180, 190, 200]})
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

print(df_scaled)
