# %% [markdown]
# # Lab 1: Data Visualization, Preprocessing & Statistical Analysis
# **Name:** Saru Bhandari  
# **Course:** Advanced Big Data and Data Mining (MSCS-634-M20)  
# **Assignment:** Lab 1

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('superstore_final_dataset.csv', encoding='latin-1')
df.head()

# %%
print(df.columns.tolist())

# %%
plt.figure(figsize=(8,5))
plt.scatter(df['Row_ID'], df['Sales'], alpha=0.5, color='steelblue')
plt.title('Row ID vs Sales')
plt.xlabel('Row ID')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# %%
df.groupby('Category')['Sales'].sum().plot(kind='bar', color='coral', figsize=(8,5))
plt.title('Total Sales by Category')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# %%
df['Sales'].hist(bins=30, color='teal', figsize=(8,5))
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.tight_layout()
plt.show()

# %%
df.boxplot(column='Sales', by='Category', figsize=(8,5))
plt.title('Sales Distribution by Category')
plt.suptitle('')
plt.tight_layout()
plt.show()

# %%
df.groupby('Region')['Sales'].sum().plot(kind='pie', autopct='%1.1f%%', figsize=(7,7))
plt.title('Sales by Region')
plt.ylabel('')
plt.tight_layout()
plt.show()

# %%
df.groupby('Order_Date')['Sales'].sum().plot(kind='line', figsize=(10,5), color='purple')
plt.title('Sales Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# %%
print("Missing values before:\n")
print(df.isnull().sum())

# %%
df['Sales'].fillna(df['Sales'].mean(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after:\n")
print(df.isnull().sum())

# %%
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1:   {Q1:.2f}")
print(f"Q3:   {Q3:.2f}")
print(f"IQR:  {IQR:.2f}")

outliers = df[(df['Sales'] < Q1 - 1.5*IQR) | (df['Sales'] > Q3 + 1.5*IQR)]
print(f"\nOutliers found: {len(outliers)}")
print(f"Shape before removing outliers: {df.shape}")

# %%
df_clean = df[(df['Sales'] >= Q1 - 1.5*IQR) & (df['Sales'] <= Q3 + 1.5*IQR)]
print(f"Shape after removing outliers: {df_clean.shape}")
df_clean.head()

# %%
print("Shape before reduction:", df_clean.shape)
df_clean.head()

# %%
df_sampled = df_clean.sample(frac=0.5, random_state=42)
df_sampled = df_sampled.drop(columns=['Row_ID', 'Postal_Code'], errors='ignore')

print("Shape after reduction:", df_sampled.shape)
df_sampled.head()

# %%
from sklearn.preprocessing import MinMaxScaler

print("Before scaling:")
print(df_sampled[['Sales']].head())

# %%
scaler = MinMaxScaler()
df_sampled = df_sampled.copy()
df_sampled['Sales_scaled'] = scaler.fit_transform(df_sampled[['Sales']])

print("After scaling:")
print(df_sampled[['Sales', 'Sales_scaled']].head())

# %%
df_sampled['Sales_category'] = pd.cut(df_sampled['Sales'], bins=3, labels=['Low', 'Medium', 'High'])
print(df_sampled['Sales_category'].value_counts())
df_sampled[['Sales', 'Sales_category']].head(10)

# %%
df.info()

# %%
df.describe()

# %%
print("Min:   ", df['Sales'].min())
print("Max:   ", df['Sales'].max())
print("Mean:  ", round(df['Sales'].mean(), 2))
print("Median:", df['Sales'].median())
print("Mode:  ", df['Sales'].mode()[0])

# %%
print("Range:   ", round(df['Sales'].max() - df['Sales'].min(), 2))
print("Q1:      ", df['Sales'].quantile(0.25))
print("Q3:      ", df['Sales'].quantile(0.75))
print("IQR:     ", round(df['Sales'].quantile(0.75) - df['Sales'].quantile(0.25), 2))
print("Variance:", round(df['Sales'].var(), 2))
print("Std Dev: ", round(df['Sales'].std(), 2))

# %%
corr_matrix = df[['Row_ID', 'Sales']].corr()
print(corr_matrix)

plt.figure(figsize=(5,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


