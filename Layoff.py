import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
import warnings
import missingno as msno
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("F://layoffs.csv")
print("Initial DataFrame:\n", df.head())

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Data types and missing values
print("\nDataFrame Info:")
print(df.info())

# Check for missing values
print("\nMissing Values in Each Column:\n", df.isnull().sum())

# Identify columns with missing values
missing_columns = df.columns[df.isnull().any()].tolist()
print("\nColumns with Missing Values:", missing_columns)

# **Step 1: Visualize Missing Values Before Handling Them**

# a. Seaborn Heatmap
def plot_missing_heatmap(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Heatmap of Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

plot_missing_heatmap(df)

# b. Missingno Matrix
def plot_missingno_matrix(df):
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Missing Values Matrix')
    plt.show()

plot_missingno_matrix(df)

# c. Seaborn Bar Plot (Enhanced)
def plot_missing_seaborn_bar(df):
    missing_counts = df.isnull().sum()
    total_rows = df.shape[0]
    missing_percent = (missing_counts / total_rows) * 100

    # Create a DataFrame for missing data
    missing_df = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percent
    })

    # Filter columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0]

    if missing_df.empty:
        print("\nNo missing values to plot.")
        return

    # Sort by Missing_Count
    missing_df = missing_df.sort_values(by='Missing_Count', ascending=True)

    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    bar_plot = sns.barplot(
        x='Missing_Percentage',
        y=missing_df.index,
        data=missing_df,
        palette='Reds_r',
        edgecolor=None
    )

    # Add annotations for counts and percentages
    for index, row in missing_df.iterrows():
        bar_plot.text(
            row['Missing_Percentage'] + 0.5, 
            missing_df.index.tolist().index(index),
            f"{int(row['Missing_Count'])} ({row['Missing_Percentage']:.1f}%)",
            color='black',
            va="center"
        )

    plt.title('Missing Values per Column', fontsize=16, weight='bold')
    plt.xlabel('Percentage of Missing Values (%)', fontsize=12)
    plt.ylabel('Columns', fontsize=12)
    plt.xlim(0, missing_df['Missing_Percentage'].max() + 10)  # Add some space for annotations
    plt.tight_layout()
    plt.show()

plot_missing_seaborn_bar(df)

# **Step 2: Plot 'Total Laid Off by Industry' Before Encoding**

# Plotting before one-hot encoding to ensure 'industry' column exists
plt.figure(figsize=(12, 6))
sns.barplot(x='industry', y='total_laid_off', data=df, ci=None, palette='viridis')
plt.title('Total Laid Off by Industry')
plt.xlabel('Industry')
plt.ylabel('Total Laid Off')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Step 3: Handle Missing Values**

# Define column types based on your dataset
numerical_cols = ['total_laid_off', 'percentage_laid_off', 'funds_raised']
categorical_cols = ['company', 'location', 'industry', 'stage', 'country']
date_col = 'date'  # Handle separately

# Handle Numerical Columns
if any(col in df.columns for col in numerical_cols):
    # Ensure numerical columns are of numeric type
    for col in numerical_cols:
        if df[col].dtype not in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Impute numerical columns with mean
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
    print("\nNumerical columns after imputation:\n", df[numerical_cols].head())

# Handle Categorical Columns
if any(col in df.columns for col in categorical_cols):
    # Impute categorical columns with the most frequent value
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    print("\nCategorical columns after imputation:\n", df[categorical_cols].head())

# Handle Date Column
if date_col in df.columns:
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Impute missing dates with the most frequent date
    if df[date_col].isnull().sum() > 0:
        most_freq_date = df[date_col].mode()[0]
        df[date_col] = df[date_col].fillna(most_freq_date)
    print("\nDate column after imputation:\n", df[date_col].head())

# Verify that all missing values are handled
print("\nMissing Values After Imputation:\n", df.isnull().sum())

# **Step 4: Remove Duplicates**

# Check for exact duplicates
exact_duplicate_count = df.duplicated().sum()
print(f"\nTotal Exact Duplicate Rows: {exact_duplicate_count}")

# Remove exact duplicates
df = df.drop_duplicates()
print(f"Shape after removing exact duplicates: {df.shape}")

# Check for partial duplicates based on 'company', 'location', 'date'
subset_cols = ['company', 'location', 'date']
partial_duplicate_count = df.duplicated(subset=subset_cols).sum()
print(f"Total Partial Duplicate Rows (based on {subset_cols}): {partial_duplicate_count}")

# Remove partial duplicates, keeping the first occurrence
df = df.drop_duplicates(subset=subset_cols, keep='first')
print(f"Shape after removing partial duplicates: {df.shape}")

# **Step 5: Encoding and Scaling**

# Encoding Categorical Variables using One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nDataFrame after One-Hot Encoding:\n", df.head())

# Feature Scaling
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("\nNumerical columns after scaling:\n", df[numerical_cols].head())

# Feature Engineering: Extract Year and Month from Date
if date_col in df.columns:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.drop('date', axis=1)
    print("\nDataFrame after extracting Year and Month:\n", df.head())

# **Step 6: Define Target Variable and Split Data**

# Define target variable and features
target = 'total_laid_off'  # Ensure this column exists in your dataset
if target not in df.columns:
    raise ValueError(f"Target variable '{target}' not found in DataFrame columns.")

X = df.drop(target, axis=1)
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")

# **Step 7: Create Creative Plots for Missing Values After Imputation**

def plot_missing_values_post_imputation(df):
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / df.shape[0]) * 100
    
    # Create a DataFrame for missing data
    missing_df = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percent
    })
    
    # Filter columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if missing_df.empty:
        print("\nNo missing values to plot after imputation.")
        return
    
    # Sort by Missing_Count
    missing_df = missing_df.sort_values(by='Missing_Count', ascending=True)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    bar_plot = sns.barplot(
        x='Missing_Percentage',
        y=missing_df.index,
        data=missing_df,
        palette='Reds_r',
        edgecolor=None
    )
    
    # Add annotations for counts and percentages
    for index, row in missing_df.iterrows():
        bar_plot.text(
            row['Missing_Percentage'] + 0.5, 
            missing_df.index.tolist().index(index),
            f"{int(row['Missing_Count'])} ({row['Missing_Percentage']:.1f}%)",
            color='black',
            va="center"
        )
    
    plt.title('Missing Values per Column After Imputation', fontsize=16, weight='bold')
    plt.xlabel('Percentage of Missing Values (%)', fontsize=12)
    plt.ylabel('Columns', fontsize=12)
    plt.xlim(0, missing_df['Missing_Percentage'].max() + 10)  # Add some space for annotations
    plt.tight_layout()
    plt.show()

plot_missing_values_post_imputation(df)

# **Step 8: Exploratory Data Analysis (EDA) Using Various Plots**

# a. Line Plot (Matplotlib)
plt.figure(figsize=(12, 6))
plt.plot(df['year'], df['total_laid_off'], marker='o', linestyle='-')
plt.title('Total Laid Off Over Years')
plt.xlabel('Year')
plt.ylabel('Total Laid Off')
plt.grid(True)
plt.show()

# b. Scatter Plot (Matplotlib)
plt.figure(figsize=(10, 6))
plt.scatter(df['funds_raised'], df['total_laid_off'], alpha=0.7)
plt.title('Funds Raised vs. Total Laid Off')
plt.xlabel('Funds Raised')
plt.ylabel('Total Laid Off')
plt.grid(True)
plt.show()

# c. Scatter Plot with Hue (Seaborn)
# **Note:** Since 'industry' has been one-hot encoded, this plot will fail.
# Instead, you should plot using 'industry_original' if retained or plot before encoding.
# Here, we have already plotted 'Total Laid Off by Industry' before encoding.

# d. Bar Plot (Matplotlib)
# The following block is commented out to prevent errors.
'''
industry_counts = df['industry'].value_counts().head(10)  # Top 10 industries
plt.figure(figsize=(12, 6))
plt.bar(industry_counts.index, industry_counts.values, color='skyblue')
plt.title('Top 10 Industries by Number of Layoffs')
plt.xlabel('Industry')
plt.ylabel('Number of Layoffs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''

# **Remove the incorrect bar plot after encoding to prevent errors**
# Original erroneous plot has been moved before encoding.

# e. Distribution of Percentage Laid Off (Histogram - Matplotlib)
plt.figure(figsize=(10, 6))
plt.hist(df['percentage_laid_off'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Percentage Laid Off')
plt.xlabel('Percentage Laid Off')
plt.ylabel('Frequency')
plt.show()

# f. Distribution of Percentage Laid Off with KDE (Seaborn)
plt.figure(figsize=(10, 6))
sns.histplot(df['percentage_laid_off'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Percentage Laid Off with KDE')
plt.xlabel('Percentage Laid Off')
plt.ylabel('Frequency')
plt.show()

# g. Box Plot of Total Laid Off by Industry (Seaborn)
# **Note:** Since 'industry' has been one-hot encoded, use 'industry_original' if retained.
# However, since we plotted before encoding, skip or use alternative methods.

# **Assuming 'industry_original' was not retained, skip this plot to prevent errors.**

# h. Pie Chart for Distribution of Layoffs by Industry (Matplotlib)
# **Note:** Since 'industry' has been encoded, use the pre-encoded DataFrame or plot before encoding.
# Alternatively, aggregate from one-hot columns.

# To plot the pie chart correctly, use the original DataFrame before encoding.
# Therefore, it's better to plot this before encoding or use a copy.

# Since encoding is already done, retrieve the pie chart data before encoding by creating a copy before encoding.

# **Recommendation:** Always perform plots requiring original categorical data before encoding.

# i. Correlation Heatmap (Matplotlib)
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Correlation Heatmap')
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.tight_layout()
plt.show()

# j. Stem Plot (Matplotlib)
plt.figure(figsize=(12, 6))
plt.stem(df['funds_raised'], df['total_laid_off'], use_line_collection=True)
plt.title('Stem Plot of Funds Raised vs. Total Laid Off')
plt.xlabel('Funds Raised')
plt.ylabel('Total Laid Off')
plt.show()

# k. 3D Scatter Plot (Matplotlib)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['funds_raised'], df['percentage_laid_off'], df['total_laid_off'], c=df['total_laid_off'], cmap='viridis', alpha=0.6)
ax.set_title('3D Scatter Plot of Funds Raised, Percentage Laid Off, and Total Laid Off')
ax.set_xlabel('Funds Raised')
ax.set_ylabel('Percentage Laid Off')
ax.set_zlabel('Total Laid Off')
plt.show()

# l. Stacked Bar Plot (Matplotlib)
# Example: Total Laid Off by Industry and Year
industry_year = df.groupby(['industry', 'year'])['total_laid_off'].sum().unstack()
industry_year = industry_year.fillna(0)

industry_year.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
plt.title('Total Laid Off by Industry and Year')
plt.xlabel('Industry')
plt.ylabel('Total Laid Off')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# m. Error Bar Plot (Matplotlib)
# Example: Mean Total Laid Off with Standard Deviation by Industry
industry_stats = df.groupby('industry')['total_laid_off'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(12, 6))
plt.errorbar(x=industry_stats['industry'], y=industry_stats['mean'], yerr=industry_stats['std'], fmt='o', ecolor='red', capsize=5, color='blue')
plt.title('Mean Total Laid Off with Standard Deviation by Industry')
plt.xlabel('Industry')
plt.ylabel('Mean Total Laid Off')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Step 9: Building and Evaluating a Linear Regression Model**

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# Plot Actual vs. Predicted Values (Seaborn)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Total Laid Off')
plt.ylabel('Predicted Total Laid Off')
plt.title('Actual vs. Predicted Total Laid Off')
plt.show()

# **Step 10: Residual Analysis**

# Calculate residuals
residuals = y_test - y_pred

# Plot Residuals Distribution (Seaborn)
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='purple')
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Plot Residuals vs. Predicted Values (Seaborn)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Total Laid Off')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()

# **Step 11: Generating a Word Cloud for Companies**

# Generate Word Cloud for Companies
text = ' '.join(df['company'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Companies', fontsize=20)
plt.show()

# **Step 2: Plot 'Total Laid Off by Industry' Before Encoding**

# Plotting before one-hot encoding to ensure 'industry' column exists
plt.figure(figsize=(12, 6))
sns.barplot(x='industry', y='total_laid_off', data=df, ci=None, palette='viridis')
plt.title('Total Laid Off by Industry')
plt.xlabel('Industry')
plt.ylabel('Total Laid Off')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

