# California County Population Analysis Script
# Creates pie chart and map showing population variation across California counties

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

def load_and_process_county_data(file_path='Input Data/hauer_county_totpop_SSPs.csv'):
    """
    Load and preprocess the California county population data
    """
    print("Loading California county population data...")
    
    # Read the data
    County_Pop = pd.read_csv(file_path)
    print(f"Loaded {len(County_Pop)} county records")
    
    # Convert county names to lowercase for merging
    County_Pop['subregion'] = County_Pop['NAME10'].str.lower()
    
    # Select only needed columns
    County_Pop = County_Pop[['averagepercent', 'subregion']]
    
    return County_Pop

def create_county_population_variation_categories(data):
    """
    Create population variation categories for counties
    """
    def categorize_variation(percent):
        if percent == 0:
            return "Population unchanged"
        elif percent < 0:
            return "Population decrease"
        else:
            return "Population increase"
    
    # Apply variation categorization
    data['Population.Variation'] = data['averagepercent'].apply(categorize_variation)
    
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
        
        # Convert county names to lowercase for merging
        ca_counties['subregion'] = ca_counties['NAME'].str.lower()
        
        if len(ca_state) > 0 and len(ca_counties) > 0:
            print("Successfully loaded US Census boundaries with counties")
            return ca_state, ca_counties
        else:
            print("Error: Failed to load California boundaries")
            raise Exception("Could not load California map data")
            
    except Exception as e:
        print(f"Error: {e}")
        raise Exception("Failed to load California boundaries. Please check your internet connection.")

def plot_california_base(ax, ca_state, ca_counties):
    """
    Plot California base map (state and county boundaries)
    """
    # Plot state with light blue fill and single black boundary
    ca_state.plot(ax=ax, color='#E6F3FF', alpha=0.5, edgecolor='black', linewidth=2)
    
    # Plot county boundaries with black lines
    ca_counties.boundary.plot(ax=ax, linewidth=1.0, edgecolor='black', alpha=0.8)
    print("Using real California boundaries with county lines")

def create_county_population_pie_chart(data, save_path=None):
    """
    Create donut chart showing county population variation distribution
    """
    print("Creating county population variation donut chart...")
    
    # Count occurrences of each category
    counts = data['Population.Variation'].value_counts()
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    
    # Define colors
    colors = ['#8DD3C7', '#FFFFB3', '#BEBADA']
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(counts.values, colors=colors, 
                                     autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*counts.sum())})', 
                                     startangle=90, wedgeprops=dict(width=0.5),  # width creates the donut hole
                                     textprops={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'},
                                     pctdistance=0.75)
    
    # Create legend for decrease and increase only
    legend_labels = []
    legend_colors = []
    legend_handles = []
    
    for i, (category, count) in enumerate(counts.items()):
        if 'decrease' in category.lower() or 'increase' in category.lower():
            legend_labels.append(category)
            legend_colors.append(colors[i])
            # Create custom legend handles
            legend_handles.append(plt.Rectangle((0,0),1,1, facecolor=colors[i]))
    
    # Add legend below the donut chart, close to the chart
    if legend_handles:
        legend = ax.legend(legend_handles, legend_labels, loc='lower center', 
                          bbox_to_anchor=(0.5, 0.10), frameon=False, 
                          fontsize=13, ncol=2, columnspacing=3.0)
        # Make legend text bold
        for text in legend.get_texts():
            text.set_fontweight('bold')
    
    # Add center text showing total count
    ax.text(0, 0, f'Total\nCounties\n{counts.sum()}', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # Remove axis
    ax.axis('equal')
    
    # Adjust layout
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches=None)
        print(f"County population donut chart saved to {save_path}")
    
    plt.show()
    return fig

def create_county_population_map(county_data, save_path=None):
    """
    Create map showing county-level population variation across California
    """
    print("Creating county population variation map...")
    
    # Load California boundaries
    ca_state, ca_counties = load_california_boundaries()
    
    # Merge county data with geographic data
    ca_counties_merged = ca_counties.merge(county_data, on='subregion', how='left')
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    ax.set_facecolor('white')
    
    # Plot California base map
    plot_california_base(ax, ca_state, ca_counties)
    
    # Create color map based on actual percentage values
    import matplotlib.colors as mcolors
    from matplotlib.cm import RdYlBu_r
    
    # Filter out NaN values for color mapping
    valid_data = ca_counties_merged.dropna(subset=['averagepercent'])
    
    if len(valid_data) > 0:
        # Create colormap
        norm = mcolors.Normalize(vmin=valid_data['averagepercent'].min(), 
                               vmax=valid_data['averagepercent'].max())
        
        # Plot counties with colors based on actual percentage values
        ca_counties_merged.plot(ax=ax, column='averagepercent', cmap='RdYlBu_r', 
                               norm=norm, alpha=0.7, edgecolor='black', linewidth=0.5,
                               missing_kwds={'color': 'lightgray'})
        
        # Add text annotations for each county showing the percentage
        for idx, row in valid_data.iterrows():
            # Get county centroid for text placement
            centroid = row.geometry.centroid
            # Format percentage
            pct_text = f"{row['averagepercent']*100:.1f}%"
            ax.annotate(pct_text, xy=(centroid.x, centroid.y), 
                       ha='center', va='center', fontsize=6, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set boundaries
    ax.set_xlim(-124.8, -113.5)  
    ax.set_ylim(31, 42.5)
    ax.axis('off')
    
    # Adjust the plot
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    ax.margins(0)
    
    # Add colorbar legend showing the percentage scale
    if len(valid_data) > 0:
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30, location='bottom', pad=0.00)
        cbar.set_label('Population Change (%)', fontsize=12, fontweight='bold')

        # Format colorbar labels as percentages
        cbar.ax.tick_params(labelsize=10)
        ticks = cbar.get_ticks()
        cbar.set_ticklabels([f'{tick*100:.1f}%' for tick in ticks])
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches=None)
        print(f"County population map saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """
    Main execution function to create both visualizations
    """
    try:
        # Load and process data
        data = load_and_process_county_data()
        data = create_county_population_variation_categories(data)
        
        print(f"\nData processing completed. Dataset contains {len(data)} county records.")
        print("\nGenerating County Population Analysis Visualizations...")
        
        # Print summary statistics
        print("\nCounty Population Variation Summary:")
        print(data['Population.Variation'].value_counts())
        
        # Create both visualizations
        pie_fig = create_county_population_pie_chart(data, 'Output Figure/county_population_pie_chart.png')
        map_fig = create_county_population_map(data, 'Output Figure/county_population_map.png')
        
        print("\nBoth visualizations created successfully!")
        print("Files saved as:")
        print("- Output Figure/county_population_pie_chart.png")
        print("- Output Figure/county_population_map.png")
        
        return data
        
    except FileNotFoundError:
        print("Error: Could not find hauer_county_totpop_SSPs.csv file.")
        print("Please make sure the file exists in 'Input Data/hauer_county_totpop_SSPs.csv'")
        return None
    
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("California County Population Analysis")
    print("=" * 50)
    print("This script creates two visualizations:")
    print("1. Pie chart showing county population variation distribution")
    print("2. Map showing county-level population variation across California")
    print("=" * 50)
    
    # Run the main function
    processed_data = main()