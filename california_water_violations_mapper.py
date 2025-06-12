# California Water System Violation and HR2W Risk Assessment Maps Script
# Creates maps showing water quality violations and HR2W risk status 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(file_path='Output Data/CWS_CA.csv'):
    """
    Load and preprocess the California water systems data
    """
    print("Loading California water systems data...")
    
    # Read the integrated water system data
    cws_ca = pd.read_csv(file_path)
    print(f"Loaded {len(cws_ca)} water system records")
    
    return cws_ca

def create_violation_categories(data):
    """
    Create violation categories for water systems
    """
    def categorize_violation(row):
        health_viol = row['Health.violation']
        monitor_viol = row['Monitoring.and.reporting.violation']
        
        if health_viol == 0 and monitor_viol == 0:
            return "No violation"
        elif health_viol > 0 and monitor_viol == 0:
            return "Health-based"
        elif health_viol == 0 and monitor_viol > 0:
            return "Monitoring & reporting"
        else:
            return "Both violations"
    
    data['Violation'] = data.apply(categorize_violation, axis=1)
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

def plot_california_base(ax, ca_state, ca_counties):
    """
    Plot California base map (state and county boundaries)
    """
    # Plot state with light blue fill and single black boundary
    ca_state.plot(ax=ax, color='#E6F3FF', alpha=0.5, edgecolor='black', linewidth=2)

    # Plot county boundaries with black lines
    ca_counties.boundary.plot(ax=ax, linewidth=1.0, edgecolor='black', alpha=0.8)

    print("Using California boundaries with county lines")

def create_violations_map(data, save_path=None):
    """
    Create map showing water system violations (Map A)
    """
    print("Creating violations map (Map A)...")
    
    # Load California boundaries
    ca_state, ca_counties = load_california_boundaries()
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    ax.set_facecolor('white')
    
    # Plot California base map
    plot_california_base(ax, ca_state, ca_counties)
    
    # Violation colors and settings
    violation_config = {
        "No violation": {"color": "#22C55E", "size": 12, "alpha": 0.5, "edge": "#15803D"},                           
        "Monitoring & reporting": {"color": "#FACC15", "size": 24, "alpha": 0.8, "edge": "#CA8A04"},       
        "Health-based": {"color": "#3B82F6", "size": 24, "alpha": 0.8, "edge": "#1D4ED8"},                 
        "Both violations": {"color": "#EF4444", "size": 30, "alpha": 0.9, "edge": "#991B1B"}                         
    }
    
    # Plot points in order (no violation first, so they're in background)
    plot_order = ["No violation", "Monitoring & reporting", "Health-based", "Both violations"]
    
    for violation_type in plot_order:
        subset = data[data['Violation'] == violation_type]
        if len(subset) > 0:
            config = violation_config[violation_type]
            ax.scatter(subset['Longitude'], subset['Latitude'],
                      c=config["color"], s=config["size"], alpha=config["alpha"],
                      label=violation_type, edgecolors=config["edge"],
                      linewidth=0.8, zorder=5)
    
    # Set boundaries 
    ax.set_xlim(-124.8, -113.5)  
    ax.set_ylim(31, 42.5)    
    ax.axis('off')
    
    # Adjust the plot area
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    ax.margins(0)
    
    # Add legend to the upper right corner
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.92), 
                      frameon=False, fontsize=14, markerscale=1.3, ncol=1)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    if save_path:
        # Save the figure
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches=None)
        print(f"Violations map saved to {save_path}")
    
    plt.show()
    return fig

def create_hr2w_map(data, save_path=None):
    """
    Create map showing HR2W risk status (Map B)
    """
    print("Creating HR2W risk assessment map (Map B)...")
    
    # Filter data with HR2W status
    data_with_hr2w = data[data['SAFER.STATUS'].notna()]
    
    # Load California boundaries
    ca_state, ca_counties = load_california_boundaries()
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    ax.set_facecolor('white')
    
    # Plot California base map
    plot_california_base(ax, ca_state, ca_counties)
    
    # HR2W colors and settings
    hr2w_config = {
        "Not At-Risk": {"color": "#22C55E", "size": 12, "alpha": 0.8, "edge": "#15803D"},         
        "Potentially At-Risk": {"color": "#3B82F6", "size": 24, "alpha": 0.85, "edge": "#1D4ED8"}, 
        "At-Risk": {"color": "#FACC15", "size": 24, "alpha": 0.85, "edge": "#CA8A04"},            
        "Failing": {"color": "#EF4444", "size": 30, "alpha": 0.9, "edge": "#991B1B"},              
        "Not Assessed": {"color": "#A855F7", "size": 12, "alpha": 0.6, "edge": "#7E22CE"}          
    }
    
    # Plot points in order
    plot_order = ["Not Assessed", "Not At-Risk", "Potentially At-Risk", "At-Risk", "Failing"]

    # Format legend labels
    legend_labels = {
        "Not At-Risk": "Not at-risk",
        "Potentially At-Risk": "Potentially at-risk",
        "At-Risk": "At-risk",
        "Failing": "Failing",
        "Not Assessed": "Not assessed"
    }
    
    for status in plot_order:
        subset = data_with_hr2w[data_with_hr2w['SAFER.STATUS'] == status]
        if len(subset) > 0:
            config = hr2w_config[status]
            ax.scatter(subset['Longitude'], subset['Latitude'],
                      c=config["color"], s=config["size"], alpha=config["alpha"],
                      edgecolors=config["edge"],
                      linewidth=0.8, zorder=5)
    
    # Set boundaries
    ax.set_xlim(-124.8, -113.5)  
    ax.set_ylim(31, 42.5)
    ax.axis('off')
    
    # Adjust the plot area
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    ax.margins(0)
    ax.margins(0)
    
    # Create custom proxy legend handles
    from matplotlib.lines import Line2D
    handles = []
    for status in plot_order:
        config = hr2w_config[status]
        label = legend_labels[status]
        handle = Line2D([0], [0],
                        marker='o', color='w',
                        markerfacecolor=config["color"],
                        markeredgecolor=config["edge"],
                        markersize=np.sqrt(config["size"]),
                        alpha=config["alpha"],
                        markeredgewidth=0.8,
                        label=label)
        handles.append(handle)

    # Add legend
    legend = ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.92),
                       frameon=False, fontsize=14, markerscale=1.3, ncol=1)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    if save_path:
        # Save the figure
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches=None)
        print(f"HR2W map saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """
    Main execution function to create both maps
    """
    try:
        # Load and process data
        data = load_and_process_data()
        data = create_violation_categories(data)
        
        print(f"\nData processing completed. Dataset contains {len(data)} records.")
        print("\nGenerating California Water Systems Maps...")
        
        # Create both maps
        violations_fig = create_violations_map(data, 'Output Figure/california_water_violations_map.png')
        hr2w_fig = create_hr2w_map(data, 'Output Figure/california_water_hr2w_map.png')
        
        print("\nBoth maps created successfully!")
        print("Files saved as:")
        print("- Output Figure/california_water_violations_map.png")
        print("- Output Figure/california_water_hr2w_map.png")
        
        return data
        
    except FileNotFoundError:
        print("Error: Could not find CWS_CA.csv file.")
        return None
    
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("California Water Systems Static Maps Generator")
    print("=" * 50)
    print("This script creates two maps:")
    print("A. Water quality violations across California")
    print("B. HR2W risk assessment status")
    print("=" * 50)
    
    # Run the main function
    processed_data = main()