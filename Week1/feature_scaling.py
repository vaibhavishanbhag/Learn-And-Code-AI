from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample Data
data = np.array([[100], [200], [300], [400], [500]])

# Standardization
scaler_standard = StandardScaler()
standardized_data = scaler_standard.fit_transform(data)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
normalized_data = scaler_minmax.fit_transform(data)

# Print Results
print("Original Data:\n", data)
print("\nStandardized Data (Z-score Normalization):\n", standardized_data)
print("\nMin-Max Scaled Data (0-1 Normalization):\n", normalized_data)
