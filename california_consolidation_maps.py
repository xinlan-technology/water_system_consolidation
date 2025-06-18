# California Water System Consolidation Analysis Maps Script
# Creates maps showing physical and managerial consolidation scenarios
# across California's community water systems

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(file_path='Output Data/CWS_CA_Case_Study_Results.csv'):
    """
    Load and preprocess the California water systems consolidation data
    """
    print("Loading California water systems consolidation data...")
    
    # Read the case study results data
    cws_ca = pd.read_csv(file_path)
    print(f"Loaded {len(cws_ca)} water system records with consolidation analysis")
    
    return cws_ca

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

def plot_california_base(ax, ca_state, ca_counties):
    """
    Plot California base map (state and county boundaries)
    """
    # Plot state with light blue fill and single black boundary
    ca_state.plot(ax=ax, color='#E6F3FF', alpha=0.5, edgecolor='black', linewidth=2)

    # Plot county boundaries with black lines
    ca_counties.boundary.plot(ax=ax, linewidth=1.0, edgecolor='black', alpha=0.8)
    print("Using California boundaries with county lines")

def create_population_analysis(data, save_path=None):
    """
    Create population distribution analysis comparing before and after consolidation
    """
    print("Creating population distribution analysis...")
    
    # Define population size categories
    def categorize_population(pop):
        if pop <= 500:
            return "0-500"
        elif pop <= 3300:
            return "501-3,300"
        elif pop <= 10000:
            return "3,301-10,000"
        elif pop <= 100000:
            return "10,001-100,000"
        else:
            return ">100,000"
    
    # 1. Before consolidation - individual systems
    before_data = data.copy()
    before_data['Size_Category'] = before_data['Population.2021'].apply(categorize_population)
    before_counts = before_data['Size_Category'].value_counts()
    print(f"Before consolidation counts: {before_counts.to_dict()}")
    
    # 2. After physical consolidation - calculate cluster populations
    physical_clusters = []
    for cluster_id in data['Physical_Cluster_Number'].unique():
        cluster_data = data[data['Physical_Cluster_Number'] == cluster_id]
        cluster_pop = cluster_data['Population.2021'].sum()
        physical_clusters.append(cluster_pop)
    
    physical_df = pd.DataFrame({'Population': physical_clusters})
    physical_df['Size_Category'] = physical_df['Population'].apply(categorize_population)
    physical_counts = physical_df['Size_Category'].value_counts()
    print(f"Physical consolidation counts: {physical_counts.to_dict()}")
    
    # 3. After managerial consolidation - calculate cluster populations
    managerial_clusters = []
    for cluster_id in data['Managerial_Cluster_Number'].unique():
        cluster_data = data[data['Managerial_Cluster_Number'] == cluster_id]
        cluster_pop = cluster_data['Population.2021'].sum()
        managerial_clusters.append(cluster_pop)
    
    managerial_df = pd.DataFrame({'Population': managerial_clusters})
    managerial_df['Size_Category'] = managerial_df['Population'].apply(categorize_population)
    managerial_counts = managerial_df['Size_Category'].value_counts()
    print(f"Managerial consolidation counts: {managerial_counts.to_dict()}")
    
    # Combine data for plotting
    categories = ["0-500", "501-3,300", "3,301-10,000", "10,001-100,000", ">100,000"]
    
    # Prepare data arrays for each scenario
    before_data_array = [before_counts.get(cat, 0) for cat in categories]
    physical_data_array = [physical_counts.get(cat, 0) for cat in categories]
    managerial_data_array = [managerial_counts.get(cat, 0) for cat in categories]
    
    print(f"Before data array: {before_data_array}")
    print(f"Physical data array: {physical_data_array}")
    print(f"Managerial data array: {managerial_data_array}")
    
    # Create the bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color scheme
    colors = ['#756bb1', '#bcbddc', '#2ca25f']
    labels = ['Before consolidation', 'Physical consolidation (1 mile threshold)', 'Managerial consolidation (10 km threshold)']
    
    # Set up the bar positions
    x = np.arange(len(categories))
    width = 0.25
    
    # Create bars for each scenario
    bars1 = ax.bar(x - width, before_data_array, width, 
                   label=labels[0], color=colors[0], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, physical_data_array, width, 
                   label=labels[1], color=colors[1], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, managerial_data_array, width, 
                   label=labels[2], color=colors[2], edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Community Water System (CWS) Size', fontsize=12)
    ax.set_ylabel('Number of CWS', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend at bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              ncol=3, frameon=False, fontsize=10)
    
    # Set background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"Population distribution chart saved to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nPopulation Distribution Summary:")
    print(f"Before consolidation: {len(before_data)} water systems")
    print(f"After physical consolidation: {len(physical_clusters)} clusters")
    print(f"After managerial consolidation: {len(managerial_clusters)} clusters")

    # Add analysis for very small systems (≤500 people)
    print(f"\nSmall Systems Analysis (≤500 people):")
    small_before = before_counts.get("0-500", 0)
    small_physical = physical_counts.get("0-500", 0) 
    small_managerial = managerial_counts.get("0-500", 0)
    
    physical_reduction = ((small_before - small_physical) / small_before * 100) if small_before > 0 else 0
    managerial_reduction = ((small_before - small_managerial) / small_before * 100) if small_before > 0 else 0
    
    print(f"Very small systems (≤500 people) before consolidation: {small_before}")
    print(f"After physical consolidation: {small_physical} (reduction: {physical_reduction:.0f}%)")
    print(f"After managerial consolidation: {small_managerial} (reduction: {managerial_reduction:.0f}%)")
    print(f"The additional managerial consolidations primarily involve very small systems,")
    print(f"potentially reducing their numbers by {managerial_reduction:.0f}% versus {physical_reduction:.0f}% through physical mergers.")
    
    return fig

def create_consolidation_map(data, consolidation_column, map_title, save_path=None):
    """
    Create map showing consolidation types for either physical or managerial scenarios
    """
    print(f"Creating {map_title.lower()}...")
    
    # Load California boundaries
    ca_state, ca_counties = load_california_boundaries()
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    ax.set_facecolor('white')
    
    # Plot California base map
    plot_california_base(ax, ca_state, ca_counties)
    
    # Consolidation type colors and settings
    consolidation_config = {
        "No_Consolidation": {"color": "#A3A3A3", "size": 12, "alpha": 0.6, "edge": "#525252"},        
        "Direct_Acquisition": {"color": "#3B82F6", "size": 24, "alpha": 0.8, "edge": "#1D4ED8"},     
        "Balanced_Merger": {"color": "#22C55E", "size": 24, "alpha": 0.8, "edge": "#15803D"},        
        "Joint_Merger": {"color": "#EF4444", "size": 30, "alpha": 0.9, "edge": "#991B1B"}            
    }
    
    # Plot points in order (no consolidation first, so they're in background)
    plot_order = ["No_Consolidation", "Direct_Acquisition", "Balanced_Merger", "Joint_Merger"]
    
    for consolidation_type in plot_order:
        subset = data[data[consolidation_column] == consolidation_type]
        if len(subset) > 0:
            config = consolidation_config[consolidation_type]
            ax.scatter(subset['Longitude'], subset['Latitude'],
                      c=config["color"], s=config["size"], alpha=config["alpha"],
                      label=consolidation_type.replace('_', ' '), edgecolors=config["edge"],
                      linewidth=0.8, zorder=5)
    
    # Set boundaries 
    ax.set_xlim(-124.8, -113.5)  
    ax.set_ylim(31, 42.5)    
    ax.axis('off')
    
    # Adjust the plot area
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    ax.margins(0)
    
    # Add legend to the upper right corner
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.92), 
                      frameon=False, fontsize=14, markerscale=1.3, ncol=1)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    if save_path:
        # Save the figure
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches=None)
        print(f"{map_title} saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """
    Main execution function to create both consolidation maps
    """
    try:
        # Load and process data
        data = load_and_process_data()
        
        print(f"\nData processing completed. Dataset contains {len(data)} records.")
        print("\nGenerating California Water Systems Consolidation Maps...")
        
        # Create both maps using the unified function
        physical_fig = create_consolidation_map(
            data, 
            'Physical_Consolidation_Type', 
            'Physical consolidation map (1 mile threshold)',
            'Output Figure/california_physical_consolidation_map.png'
        )
        
        managerial_fig = create_consolidation_map(
            data, 
            'Managerial_Consolidation_Type', 
            'Managerial consolidation map (10 km threshold)',
            'Output Figure/california_managerial_consolidation_map.png'
        )
        
        # Create population distribution analysis
        population_fig = create_population_analysis(
            data,
            'Output Figure/california_population_distribution.png'
        )
        
        print("\nAll visualizations created successfully!")
        print("Files saved as:")
        print("- Output Figure/california_physical_consolidation_map.png")
        print("- Output Figure/california_managerial_consolidation_map.png")
        print("- Output Figure/california_population_distribution.png")
        
        return data
        
    except FileNotFoundError:
        print("Error: Could not find CWS_CA_Case_Study_Results.csv file.")
        return None
    
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("California Water Systems Consolidation Maps Generator")
    print("=" * 60)
    print("This script creates two consolidation analysis maps:")
    print("A. Physical Consolidation (1 mile threshold)")
    print("B. Managerial Consolidation (10 km threshold)")
    print("=" * 60)

    print("California Water Systems Population Distribution Analysis Before and After Consolidation")
    print("=" * 60)
    print("This script creates a population distribution analysis comparing before and after consolidation")
    print("=" * 60)
    
    # Run the main function
    processed_data = main()