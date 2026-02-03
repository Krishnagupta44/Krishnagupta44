import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# -----------------------------
# Create output folder
# -----------------------------
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("library_dataset.csv")

# -----------------------------
# 1. Data Cleaning: Smoothing by Binning
# -----------------------------
df['Price_Bin'] = pd.cut(df['Book_Price'], bins=5)

# Mean smoothing
df['Mean_Smooth'] = df.groupby('Price_Bin')['Book_Price'].transform('mean')

plt.figure()
plt.plot(df['Book_Price'], label='Original', marker='o')
plt.plot(df['Mean_Smooth'], label='Mean Smoothed', marker='x')
plt.legend()
plt.title("Smoothing by Mean")
plt.savefig(f"{output_dir}/binning_mean.png")
plt.close()

# Median smoothing
df['Median_Smooth'] = df.groupby('Price_Bin')['Book_Price'].transform('median')

plt.figure()
plt.plot(df['Book_Price'], label='Original', marker='o')
plt.plot(df['Median_Smooth'], label='Median Smoothed', marker='x')
plt.legend()
plt.title("Smoothing by Median")
plt.savefig(f"{output_dir}/binning_median.png")
plt.close()

# Boundary smoothing
def boundary_smoothing(series):
    min_val = series.min()
    max_val = series.max()
    return series.apply(lambda x: min_val if x - min_val < max_val - x else max_val)

df['Boundary_Smooth'] = df.groupby('Price_Bin')['Book_Price'].transform(boundary_smoothing)

plt.figure()
plt.plot(df['Book_Price'], label='Original', marker='o')
plt.plot(df['Boundary_Smooth'], label='Boundary Smoothed', marker='x')
plt.legend()
plt.title("Smoothing by Boundaries")
plt.savefig(f"{output_dir}/binning_boundary.png")
plt.close()

# -----------------------------
# 2. Remove Redundancy
# -----------------------------
df_clean = df.drop_duplicates()

# Save redundancy comparison image
plt.figure()
plt.bar(['Before', 'After'], [len(df), len(df_clean)])
plt.title("Redundancy Removal")
plt.ylabel("Number of Records")
plt.savefig(f"{output_dir}/redundancy_removal.png")
plt.close()

# -----------------------------
# 3. Normalization
# -----------------------------
# Min-Max normalization
df_clean['MinMax_Price'] = (
    (df_clean['Book_Price'] - df_clean['Book_Price'].min()) /
    (df_clean['Book_Price'].max() - df_clean['Book_Price'].min())
)

plt.figure()
plt.plot(df_clean['MinMax_Price'], color='green')
plt.title("Min-Max Normalization")
plt.savefig(f"{output_dir}/minmax_normalization.png")
plt.close()

# Z-score normalization
df_clean['ZScore_Price'] = stats.zscore(df_clean['Book_Price'])

plt.figure()
plt.plot(df_clean['ZScore_Price'], color='purple')
plt.title("Z-Score Normalization")
plt.savefig(f"{output_dir}/zscore_normalization.png")
plt.close()

# -----------------------------
# 4. Histogram Analysis
# -----------------------------
plt.figure()
plt.hist(df_clean['Book_Price'], bins=10, edgecolor='black')
plt.xlabel("Book Price")
plt.ylabel("Frequency")
plt.title("Histogram of Book Prices")
plt.savefig(f"{output_dir}/book_price.png")
plt.close()

# -----------------------------
# 5. Outlier Detection
# -----------------------------
plt.figure()
plt.boxplot(df_clean['Book_Price'], vert=False)
plt.title("Outlier Detection using Boxplot")
plt.savefig(f"{output_dir}/outliers_boxplot.png")
plt.close()

print("All operations completed. Images saved in 'output_images' folder.")
