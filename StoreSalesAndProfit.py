import pandas as pd
import matplotlib.pyplot as plt

#CSV file location
file_path = r'Intermediate Store Sales and Profit Analysis\Sample - Superstore.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')#Convert the field Order Date to datetime
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')#Convert the field Ship Date  to datetime

#Category Analysis
category_analysis = df.groupby(['Category', 'Sub-Category']).agg({#Grouping data by category and sub-Category to analyze sales and profit
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values(by='Sales', ascending=False)
print("Sales and profit by category and subcategory")#Display the analysis
print(category_analysis)

#Region Analysis
region_analysis = df.groupby('Region').agg({#Grouping data by Region to analyze sales and profit
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values(by='Sales', ascending=False)
print("\nSales and profit by region")
print(region_analysis)


#Grouping data by month to analyze sales and profit trends
df['Order month'] = df['Order Date'].dt.to_period('M')
monthly_trends = df.groupby('Order month').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

#Plotting the sales and profit trends over time
plt.figure(figsize=(12, 6))
plt.plot(monthly_trends.index.astype(str), monthly_trends['Sales'], label='Sales', color='black')
plt.plot(monthly_trends.index.astype(str), monthly_trends['Profit'], label='Profit', marker='o', color='pink')
plt.xticks(rotation=45)
plt.title('Monthly sales and profit trends')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True)
plt.show()
