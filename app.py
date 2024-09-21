# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression

# Set the page configuration (optional)
st.set_page_config(page_title="Rupa & Co. Sales Potential", layout="wide")

# Title and Description
st.title('Rupa & Co. Sales Potential')
st.write('''
Welcome to the **Rupa & Co. Sales Potential** app!

This application allows you to explore the sales potential of different Indian cities for various product types.

**Instructions:**

1. **Enter a City Name** in the sidebar.
2. View the **Sales Potential** for **Luxury Goods**, **Standard Goods**, and **Essential Goods**.
3. Explore the **graphs** and **descriptions** to understand the insights.

The app uses a simple machine learning model to estimate sales potential based on city demographics.
''')

# Sidebar for User Input
st.sidebar.header('Enter City Name')
city_name_input = st.sidebar.text_input('City Name', '')

# Load the Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('city.csv', dtype={'Population (2011)': str})
    return df

df = load_data()

# Data Cleaning and Preprocessing
def clean_data(df):
    # Remove commas and convert 'Population (2011)' to integer
    df['Population (2011)'] = df['Population (2011)'].str.replace(',', '', regex=False)
    df['Population (2011)'] = pd.to_numeric(df['Population (2011)'], errors='coerce')
    df = df.dropna(subset=['Population (2011)'])
    df['Population (2011)'] = df['Population (2011)'].astype(int)
    
    # Handle 'Area (sq km)' if available
    if 'Area (sq km)' in df.columns:
        df['Area (sq km)'] = df['Area (sq km)'].str.replace(',', '', regex=False)
        df['Area (sq km)'] = pd.to_numeric(df['Area (sq km)'], errors='coerce')
        df = df.dropna(subset=['Area (sq km)'])
        df['Area (sq km)'] = df['Area (sq km)'].astype(float)
    else:
        # Estimate area based on average population density
        avg_density = 5000  # Adjust as needed
        df['Area (sq km)'] = df['Population (2011)'] / avg_density

    # Calculate Population Density
    df['Population Density'] = df['Population (2011)'] / df['Area (sq km)']

    # Handle 'GDP (INR crores)' if available
    if 'GDP (INR crores)' in df.columns:
        df['GDP (INR crores)'] = df['GDP (INR crores)'].str.replace(',', '', regex=False)
        df['GDP (INR crores)'] = pd.to_numeric(df['GDP (INR crores)'], errors='coerce')
        df = df.dropna(subset=['GDP (INR crores)'])
        df['GDP (INR crores)'] = df['GDP (INR crores)'].astype(float)
    else:
        # Generate synthetic GDP data
        df['GDP (INR crores)'] = df['Population (2011)'] * np.random.uniform(0.1, 0.5)

    # Calculate GDP per Capita
    df['GDP per Capita'] = (df['GDP (INR crores)'] * 1e7) / df['Population (2011)']  # Convert crores to units

    return df

df = clean_data(df)

# Simulate Sales Potential
def simulate_sales_potential(df):
    # Calculate Sales Potential for each Product Type
    def calculate_sales_potential(row, product_type):
        if product_type == 'Luxury Goods':
            alpha = 0.5
            beta = 0.5
        elif product_type == 'Standard Goods':
            alpha = 0.7
            beta = 0.3
        else:  # Essential Goods
            alpha = 0.9
            beta = 0.1
        potential = alpha * row['Population (2011)'] + beta * row['GDP per Capita']
        return potential

    df['Luxury Goods Potential'] = df.apply(lambda row: calculate_sales_potential(row, 'Luxury Goods'), axis=1)
    df['Standard Goods Potential'] = df.apply(lambda row: calculate_sales_potential(row, 'Standard Goods'), axis=1)
    df['Essential Goods Potential'] = df.apply(lambda row: calculate_sales_potential(row, 'Essential Goods'), axis=1)
    
    return df

df = simulate_sales_potential(df)

# Machine Learning Model Description
st.sidebar.markdown('''
---
**About the ML Model:**

- The model estimates sales potential using a simple formula combining **population** and **GDP per capita**.
- Different weights (**alpha** and **beta**) are applied for each product type:
  - **Luxury Goods**: More influenced by GDP per capita.
  - **Essential Goods**: More influenced by population.
- This approach helps identify cities with higher demand potential for different products.

---
''')

# Main App Logic
if city_name_input:
    city_name_input = city_name_input.strip().title()
    # Filter the dataframe for the entered city name
    city_data = df[df['Name of City'].str.title() == city_name_input]
    if not city_data.empty:
        st.subheader(f"Sales Potential for **{city_name_input}**")

        # Display the sales potential for different product types
        potentials = city_data[['Luxury Goods Potential', 'Standard Goods Potential', 'Essential Goods Potential']].iloc[0]
        st.write(f"**Luxury Goods Potential:** {potentials['Luxury Goods Potential']:.2f}")
        st.write(f"**Standard Goods Potential:** {potentials['Standard Goods Potential']:.2f}")
        st.write(f"**Essential Goods Potential:** {potentials['Essential Goods Potential']:.2f}")

        # Display additional information
        st.subheader('City Details')
        city_info = city_data[['Name of City', 'State', 'Population (2011)', 'GDP per Capita', 'Population Density']].iloc[0]
        st.write(city_info)

        # Plotting the sales potential by product type
        st.subheader('Sales Potential by Product Type')
        st.write('This bar chart shows the estimated sales potential for each product type in the selected city.')

        fig, ax = plt.subplots()
        product_types = ['Luxury Goods', 'Standard Goods', 'Essential Goods']
        values = [potentials['Luxury Goods Potential'], potentials['Standard Goods Potential'], potentials['Essential Goods Potential']]
        # Use valid color codes
        ax.bar(product_types, values, color=['#FFD700', '#C0C0C0', '#CD7F32'])
        ax.set_ylabel('Sales Potential')
        ax.set_xlabel('Product Type')
        ax.set_title('Sales Potential for Different Product Types')
        st.pyplot(fig)

        # Additional Visualization: Compare with Average Sales Potential
        st.subheader('Comparison with Average Sales Potential')
        st.write('This chart compares the city\'s sales potential with the average across all cities in the dataset.')

        avg_potentials = df[['Luxury Goods Potential', 'Standard Goods Potential', 'Essential Goods Potential']].mean()
        comparison_df = pd.DataFrame({
            'Product Type': ['Luxury Goods', 'Standard Goods', 'Essential Goods'],
            'City Potential': values,
            'Average Potential': avg_potentials.values
        })
        # Plot comparison
        fig2, ax2 = plt.subplots()
        index = np.arange(len(comparison_df['Product Type']))
        bar_width = 0.35
        opacity = 0.8

        rects1 = ax2.bar(index, comparison_df['City Potential'], bar_width,
                         alpha=opacity, color='b', label=city_name_input)

        rects2 = ax2.bar(index + bar_width, comparison_df['Average Potential'], bar_width,
                         alpha=opacity, color='g', label='Average')

        ax2.set_xlabel('Product Type')
        ax2.set_ylabel('Sales Potential')
        ax2.set_title('City vs. Average Sales Potential')
        ax2.set_xticks(index + bar_width / 2)
        ax2.set_xticklabels(comparison_df['Product Type'])
        ax2.legend()

        # Add labels on top of bars
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)

        st.pyplot(fig2)

    else:
        st.error(f"City '{city_name_input}' not found in the dataset.")
else:
    st.write('Please enter a city name in the sidebar to view its sales potential.')

# Optionally, display the list of available cities
if st.sidebar.checkbox('Show Available Cities'):
    st.subheader('List of Cities in the Dataset')
    st.write(df['Name of City'].unique())
