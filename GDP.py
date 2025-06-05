import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Sample GDP data for Kenya (in billion USD)
# Note: For actual analysis, use real data from World Bank/IMF/KNBS
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'GDP': [64.22, 70.53, 79.26, 87.91, 95.50, 98.84, 110.35, 118.14, 125.32, 133.91, 142.80],
    'Agriculture': [25.1, 26.5, 27.8, 28.9, 29.5, 29.8, 30.2, 30.8, 31.5, 32.1, 32.8],
    'Industry': [17.2, 18.6, 20.1, 21.8, 23.1, 22.9, 24.5, 25.7, 27.1, 28.6, 30.2],
    'Services': [42.7, 45.4, 48.9, 52.3, 56.2, 57.4, 60.8, 63.9, 67.2, 71.1, 75.3]
}

df = pd.DataFrame(data)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index.year, y='GDP', marker='o')
plt.title("Kenya's GDP Growth (2015-2025)", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('GDP (Billion USD)', fontsize=12)
plt.grid(True)
plt.xticks(df.index.year)
plt.tight_layout()
plt.show()

# Calculate sector contributions
sectors = ['Agriculture', 'Industry', 'Services']
df_sectors = df[sectors].div(df['GDP'], axis=0) * 100

plt.figure(figsize=(12, 6))
df_sectors.plot(kind='area', stacked=True, alpha=0.8)
plt.title("Sector Contribution to Kenya's GDP (%)", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage Contribution', fontsize=12)
plt.legend(title='Sectors', bbox_to_anchor=(1.05, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate annual growth rates
df['GDP_Growth_Rate'] = df['GDP'].pct_change() * 100

plt.figure(figsize=(12, 6))
bars = plt.bar(df.index.year, df['GDP_Growth_Rate'], color='skyblue')
plt.title("Kenya's Annual GDP Growth Rate (%)", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Growth Rate (%)', fontsize=12)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')

plt.grid(axis='y')
plt.xticks(df.index.year)
plt.tight_layout()
plt.show()

# Decompose GDP time series
result = seasonal_decompose(df['GDP'], model='additive', period=1)

plt.figure(figsize=(12, 8))
result.plot()
plt.suptitle("Time Series Decomposition of Kenya's GDP", y=1.02)
plt.tight_layout()
plt.show()

# Prepare data for forecasting
X = np.array(df.index.year).reshape(-1, 1)
y = df['GDP'].values

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Forecast future years
future_years = np.array(range(2026, 2031)).reshape(-1, 1)
forecast = model.predict(future_years)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'GDP': forecast
})

# Plot historical and forecasted GDP
plt.figure(figsize=(12, 6))
plt.plot(df.index.year, df['GDP'], 'b-o', label='Historical GDP')
plt.plot(forecast_df['Year'], forecast_df['GDP'], 'r--o', label='Forecasted GDP')
plt.title("Kenya's GDP Historical Trend and Forecast (2026-2030)", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('GDP (Billion USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(np.append(df.index.year, forecast_df['Year']))
plt.tight_layout()
plt.show()

# Calculate correlation matrix
corr_matrix = df[sectors].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Between Economic Sectors", fontsize=16)
plt.tight_layout()
plt.show()# Calculate correlation matrix
corr_matrix = df[sectors].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Between Economic Sectors", fontsize=16)
plt.tight_layout()
plt.show()