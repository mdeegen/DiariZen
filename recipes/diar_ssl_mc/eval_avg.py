import re
import numpy as np

# Input string
data = "                 7.6 (21.3 / 2.6)    & 4.6 (20.0 / 3.0)& 10.2 (23.6 / 2.5) &13.4 (24.5 / 4.2)                          "

# Extract all number triplets using regex
matches = re.findall(r'([\d.]+)\s*\(([\d.]+)\s*/\s*([\d.]+)\)', data)

# Convert to float and store in a NumPy array
values = np.array([[float(a), float(b), float(c)] for a, b, c in matches])

# Compute column-wise averages
averages = values.sum(axis=0) / 4

print(f"Average (main / first / second): {averages[0]:.2f} ({averages[1]:.2f} / {averages[2]:.2f})")
