import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load sample dataset
data = {
    'Age': [25, 30, 35, None, 40],
    'Salary': [50000, 60000, None, 80000, 90000],
    'Country': ['USA', 'Canada', 'USA', 'India', 'Canada']
}

df = pd.DataFrame(data)

# Handling missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encoding categorical variables
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Country']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Concatenating the processed data
df = df.drop(columns=['Country']).reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

# Scaling numerical data
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print(df.head())
