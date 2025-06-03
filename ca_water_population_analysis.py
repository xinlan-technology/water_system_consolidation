# California Water Systems Population Analysis Script
# Creates bar chart and map showing population variation across water systems

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(file_path='Output Data/CWS_CA.csv'):
    """
    Load and preprocess the California water systems data
    """
    print("Loading California water systems data...")
    
    # Read the data
    CWS_CA = pd.read_csv(file_path)
    print(f"Loaded {len(CWS_CA)} water system records")
    
    return CWS_CA

def create_size_categories(data):
    """
    Create water system size categories based on population served
    """
    def categorize_size(population):
        if 0 < population <= 500:
            return "0-500"
        elif 501 <= population <= 3300:
            return "501-3300"
        elif 3301 <= population <= 10000:
            return "3301-10000"
        elif 10001 <= population <= 100000:
            return "10001-100000"
        else:
            return ">100000"
    
    # Apply size categorization
    data['Size'] = data['Population.2021'].apply(categorize_size)
    
    # Set proper order for categories
    size_order = ["0-500", "501-3300", "3301-10000", "10001-100000", ">100000"]
    data['Size'] = pd.Categorical(data['Size'], categories=size_order, ordered=True)
    
    return data

def create_population_variation_categories(data):
    """
    Create population variation categories
    """
    def categorize_variation(change):
        if change == 0:
            return "Population unchanged"
        elif change < 0:
            return "Population decrease"
        else:
            return "Population increase"
    
    # Apply variation categorization
    data['Population.Variation'] = data['Population.Change'].apply(categorize_variation)
    
    # Set proper order for categories
    variation_order = ["Population decrease", "Population unchanged", "Population increase"]
    data['Population.Variation'] = pd.Categorical(data['Population.Variation'], 
                                                 categories=variation_order, ordered=True)
    
    return data

def load_california_boundaries():
    """
    Load California state and county boundaries using US Census Bureau data
    """
    try:
        print("Loading California geographic boundaries...")
        
        # Use US Census Bureau data
        states_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
        states_gdf = gpd.read_file(states_url)
        ca_state = states_gdf[states_gdf['NAME'] == 'California']
        
        counties_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip"
        counties_gdf = gpd.read_file(counties_url)
        ca_counties = counties_gdf[counties_gdf['STATEFP'] == '06']
        
        if len(ca_state) > 0 and len(ca_counties) > 0:
            print("Successfully loaded US Census boundaries with counties")
            return ca_state, ca_counties
        else:
            print("Error: Failed to load California boundaries")
            raise Exception("Could not load California map data")
            
    except Exception as e:
        print(f"Error: {e}")
        raise Exception("Failed to load California boundaries. Please check your internet connection.")

def create_population_variation_bar_chart(data, save_path=None):
    """
    Create stacked bar chart showing population variation by water system size
    """
    print("Creating population variation stacked bar chart...")
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    
    # Define colors
    colors = ['#7DD3FC', '#86EFAC', '#FDE68A', '#FECACA', '#C4B5FD'] 
    
    # Prepare data for stacked bar chart
    variation_categories = ["Population decrease", "Population unchanged", "Population increase"]
    size_categories = ["0-500", "501-3300", "3301-10000", "10001-100000", ">100000"]
    
    # Create a crosstab to get counts for each combination
    crosstab_data = pd.crosstab(data['Population.Variation'], data['Size'])
    
    # Reorder columns to match size categories order
    crosstab_data = crosstab_data.reindex(columns=size_categories, fill_value=0)
    crosstab_data = crosstab_data.reindex(index=variation_categories, fill_value=0)
    
    # Create stacked bar chart
    bottom = np.zeros(len(variation_categories))
    
    for i, size_cat in enumerate(size_categories):
        values = crosstab_data[size_cat].values
        ax.bar(variation_categories, values, bottom=bottom, 
               label=size_cat, color=colors[i], width=0.6)
        bottom += values
    
    # Customize the plot
    ax.set_xlabel('Population Served Variation', fontsize=14)
    ax.set_ylabel('Quantity', fontsize=14)
    
    # Remove grid and background
    ax.grid(False)
    ax.set_facecolor('white')
    
    # Add legend at bottom
    ax.legend(title='Community Water System Size', loc='upper center', 
             bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=12)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Adjust layout to match map format
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Stacked bar chart saved to {save_path}")
    
    plt.show()
    return fig

def plot_california_base(ax, ca_state, ca_counties):
    """
    Plot California base map (state and county boundaries)
    """
    # Plot state with light blue fill and single black boundary
    ca_state.plot(ax=ax, color='#E6F3FF', alpha=0.5, edgecolor='black', linewidth=2)
    
    # Plot county boundaries with black lines
    ca_counties.boundary.plot(ax=ax, linewidth=1.0, edgecolor='black', alpha=0.8)
    print("Using real California boundaries with county lines")

def create_population_variation_map(data, save_path=None):
    """
    Create map showing population variation across California
    """
    print("Creating population variation map...")
    
    # Load California boundaries
    ca_state, ca_counties = load_california_boundaries()
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    ax.set_facecolor('white')
    
    # Plot California base map
    plot_california_base(ax, ca_state, ca_counties)
    
    # Define colors and sizes for population variation
    variation_config = {
        "Population increase": {"color": "#3B82F6", "size": 24, "alpha": 0.8, "edge": "#1D4ED8"},       # 蓝色
        "Population unchanged": {"color": "#22C55E", "size": 20, "alpha": 0.7, "edge": "#15803D"},      # 绿色
        "Population decrease": {"color": "#EF4444", "size": 28, "alpha": 0.8, "edge": "#991B1B"}       # 红色
    }
    
    # Plot points in order (increase first, then unchange, then decrease)
    plot_order = ["Population increase", "Population unchanged", "Population decrease"]
    
    for variation in plot_order:
        subset = data[data['Population.Variation'] == variation]
        if len(subset) > 0:
            # Add small random jitter to avoid overlapping points
            jitter_x = np.random.normal(0, 0.01, len(subset))
            jitter_y = np.random.normal(0, 0.01, len(subset))
            
            config = variation_config[variation]
            ax.scatter(subset['Longitude'] + jitter_x, 
                      subset['Latitude'] + jitter_y,
                      c=config["color"], s=config["size"], alpha=config["alpha"],
                      label=variation, edgecolors=config["edge"], 
                      linewidth=0.8, zorder=5)
    
    # Set boundaries
    ax.set_xlim(-124.8, -113.5)  
    ax.set_ylim(31, 42.5)
    ax.axis('off')
    
    # Adjust the plot area
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    ax.margins(0)
    
    # Add legend to the upper right corner
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.92), 
                      frameon=False, fontsize=14, markerscale=1.3, ncol=1)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    if save_path:
        # Save the figure
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches=None)
        print(f"Population variation map saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """
    Main execution function to create both visualizations
    """
    try:
        # Load and process data
        data = load_and_process_data()
        data = create_size_categories(data)
        data = create_population_variation_categories(data)
        
        print(f"\nData processing completed. Dataset contains {len(data)} records.")
        print("\nGenerating Population Analysis Visualizations...")
        
        # Print summary statistics
        print("\nPopulation Variation Summary:")
        print(data['Population.Variation'].value_counts())
        print("\nWater System Size Summary:")
        print(data['Size'].value_counts())
        
        # Create both visualizations
        bar_fig = create_population_variation_bar_chart(data, 'Output Figure/population_variation_bar_chart.png')
        map_fig = create_population_variation_map(data, 'Output Figure/population_variation_map.png')
        
        print("\nBoth visualizations created successfully!")
        print("Files saved as:")
        print("- Output Figure/population_variation_bar_chart.png")
        print("- Output Figure/population_variation_map.png")
        
        return data
        
    except FileNotFoundError:
        print("Error: Could not find CWS_CA.csv file.")
        print("Please make sure the file exists in 'Output Data/CWS_CA.csv'")
        return None
    
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("California Water Systems Population Analysis")
    print("=" * 50)
    print("This script creates two visualizations:")
    print("1. Bar chart showing population variation by system size")
    print("2. Map showing population variation across California")
    print("=" * 50)
    
    # Run the main function
    processed_data = main()