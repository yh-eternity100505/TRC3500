import pandas as pd
import numpy as np
import json

def get_regression_from_excel(file_path, sheet_name, x_column_name, y_column_name):
    """
    Reads an Excel file and computes linear regression (y = mx + c).
    """
    # 1. Load the Excel file into a DataFrame
    # If your data is on the first sheet, you can omit sheet_name
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 2. Extract the columns
    # We use .values to get numpy arrays, and dropna() to handle empty cells
    data = df[[x_column_name, y_column_name]].dropna()
    x = data[x_column_name].values
    y = data[y_column_name].values

    # 3. Perform Linear Regression
    # np.polyfit(x, y, 1) returns [slope, intercept]
    m, c = np.polyfit(x, y, 1)

    return m, c

# --- Example Usage ---
# Change 'data.xlsx' to your actual file name
file = 'sensor_data.xlsx' 

# For ADC 1
m1, c1 = get_regression_from_excel(file, sheet_name='Sheet1', x_column_name='ml', y_column_name='ADC 1')
print(f"ADC 1 Equation: y = {m1:.4f}x + {c1:.4f}")

calibration_data = {
    "ADC_1": {"m": -41.25, "c": 3650.5}
}

# Save to a file
with open('params.txt', 'w') as f:
    f.write(f"{m1}\n{c1}")