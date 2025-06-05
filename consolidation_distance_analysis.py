# Script to visualize water system distance and consolidation analysis
# Creates two plots: nearest neighbor distances and consolidation sensitivity across distance thresholds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def water_systems_nearest_neighbor_analysis():
    """
    Calculate and visualize nearest neighbor distances for California water systems
    """
    
    # Load data
    print("Loading data...")
    try:
        # Load CWS data
        cws_ca = pd.read_csv('Output Data/CWS_CA.csv')
        print(f"Loaded {len(cws_ca)} water systems")
        
        # Load distance matrix
        data = np.load('Output Data/PWSID_Distance_Matrix_km.npz')
        distance_matrix_km = data['distance_matrix']
        pwsids = data['pwsids']
        print(f"Loaded distance matrix: {distance_matrix_km.shape[0]} × {distance_matrix_km.shape[1]} systems")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate nearest neighbor distances
    print("Calculating nearest neighbor distances...")
    
    # Create a copy of distance matrix and set diagonal to infinity to exclude self-distances
    dist_matrix = distance_matrix_km.copy()
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Find minimum distance for each system (nearest neighbor)
    nearest_distances_km = np.min(dist_matrix, axis=1)
    nearest_distances_miles = nearest_distances_km * 0.621371  # Convert km to miles
    
    # Sort distances for plotting
    sorted_distances_km = np.sort(nearest_distances_km)
    sorted_distances_miles = np.sort(nearest_distances_miles)
    
    # Calculate statistics
    stats_km = {
        'min': nearest_distances_km.min(),
        'max': nearest_distances_km.max(),
        'median': np.median(nearest_distances_km),
        'mean': nearest_distances_km.mean()
    }
    
    stats_miles = {
        'min': nearest_distances_miles.min(),
        'max': nearest_distances_miles.max(),
        'median': np.median(nearest_distances_miles),
        'mean': nearest_distances_miles.mean()
    }
    
    # Print statistics
    print("\n" + "="*60)
    print("NEAREST NEIGHBOR DISTANCE STATISTICS")
    print("="*60)
    print(f"Total systems analyzed: {len(nearest_distances_km)}")
    print(f"\nDistance Statistics (km):")
    print(f"  Minimum:  {stats_km['min']:.2f} km")
    print(f"  Maximum:  {stats_km['max']:.2f} km")
    print(f"  Median:   {stats_km['median']:.2f} km")
    print(f"  Mean:     {stats_km['mean']:.2f} km")
    
    print(f"\nDistance Statistics (miles):")
    print(f"  Minimum:  {stats_miles['min']:.2f} miles")
    print(f"  Maximum:  {stats_miles['max']:.2f} miles")
    print(f"  Median:   {stats_miles['median']:.2f} miles")
    print(f"  Mean:     {stats_miles['mean']:.2f} miles")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Create system numbers for x-axis
    system_numbers = np.arange(1, len(sorted_distances_km) + 1)
    
    # Plot km distances on left y-axis
    color_km = 'tab:blue'
    ax1.set_xlabel('System Number (Sorted by Distance)', fontsize=12)
    ax1.set_ylabel('Distance (km)', color=color_km, fontsize=12)
    line1 = ax1.plot(system_numbers, sorted_distances_km, color='black', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_km)
    ax1.grid(True, alpha=0.3)
    
    # Set axis limits to start from 0
    ax1.set_xlim(0, len(sorted_distances_km)+50)
    ax1.set_ylim(0, sorted_distances_km.max() * 1.05)
    
    # Create second y-axis for miles
    ax2 = ax1.twinx()
    color_miles = 'tab:red'
    ax2.set_ylabel('Distance (miles)', color=color_miles, fontsize=12)
    line2 = ax2.plot(system_numbers, sorted_distances_miles, color='black', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_miles)
    
    # Set right y-axis to also start from 0
    ax2.set_ylim(0, sorted_distances_miles.max() * 1.05)
    
    plt.tight_layout()
    plt.savefig('Output Figure/Water_Systems_Nearest_Neighbor_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: Output Figure/Water_Systems_Nearest_Neighbor_Analysis.png")
    
    # Create consolidation sensitivity plot
    create_consolidation_sensitivity_plot()
    
    return nearest_distances_km, nearest_distances_miles

# Script to visualize water system distance and consolidation analysis
# Creates two plots: nearest neighbor distances and consolidation sensitivity across distance thresholds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def water_systems_nearest_neighbor_analysis():
    """
    Calculate and visualize nearest neighbor distances for California water systems
    """
    
    # Load data
    print("Loading data...")
    try:
        # Load CWS data
        cws_ca = pd.read_csv('Output Data/CWS_CA.csv')
        print(f"Loaded {len(cws_ca)} water systems")
        
        # Load distance matrix
        data = np.load('Output Data/PWSID_Distance_Matrix_km.npz')
        distance_matrix_km = data['distance_matrix']
        pwsids = data['pwsids']
        print(f"Loaded distance matrix: {distance_matrix_km.shape[0]} × {distance_matrix_km.shape[1]} systems")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate nearest neighbor distances
    print("Calculating nearest neighbor distances...")
    
    # Create a copy of distance matrix and set diagonal to infinity to exclude self-distances
    dist_matrix = distance_matrix_km.copy()
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Find minimum distance for each system (nearest neighbor)
    nearest_distances_km = np.min(dist_matrix, axis=1)
    nearest_distances_miles = nearest_distances_km * 0.621371  # Convert km to miles
    
    # Sort distances for plotting
    sorted_distances_km = np.sort(nearest_distances_km)
    sorted_distances_miles = np.sort(nearest_distances_miles)
    
    # Calculate statistics
    stats_km = {
        'min': nearest_distances_km.min(),
        'max': nearest_distances_km.max(),
        'median': np.median(nearest_distances_km),
        'mean': nearest_distances_km.mean()
    }
    
    stats_miles = {
        'min': nearest_distances_miles.min(),
        'max': nearest_distances_miles.max(),
        'median': np.median(nearest_distances_miles),
        'mean': nearest_distances_miles.mean()
    }
    
    # Print statistics
    print("\n" + "="*60)
    print("NEAREST NEIGHBOR DISTANCE STATISTICS")
    print("="*60)
    print(f"Total systems analyzed: {len(nearest_distances_km)}")
    print(f"\nDistance Statistics (km):")
    print(f"  Minimum:  {stats_km['min']:.2f} km")
    print(f"  Maximum:  {stats_km['max']:.2f} km")
    print(f"  Median:   {stats_km['median']:.2f} km")
    print(f"  Mean:     {stats_km['mean']:.2f} km")
    
    print(f"\nDistance Statistics (miles):")
    print(f"  Minimum:  {stats_miles['min']:.2f} miles")
    print(f"  Maximum:  {stats_miles['max']:.2f} miles")
    print(f"  Median:   {stats_miles['median']:.2f} miles")
    print(f"  Mean:     {stats_miles['mean']:.2f} miles")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Create system numbers for x-axis
    system_numbers = np.arange(1, len(sorted_distances_km) + 1)
    
    # Plot km distances on left y-axis
    color_km = 'tab:blue'
    ax1.set_xlabel('System Number (Sorted by Distance)', fontsize=12)
    ax1.set_ylabel('Distance (km)', color=color_km, fontsize=12)
    line1 = ax1.plot(system_numbers, sorted_distances_km, color='black', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_km)
    ax1.grid(True, alpha=0.3)
    
    # Set axis limits to start from 0
    ax1.set_xlim(0, len(sorted_distances_km))
    ax1.set_ylim(0, sorted_distances_km.max() * 1.05)
    
    # Create second y-axis for miles
    ax2 = ax1.twinx()
    color_miles = 'tab:red'
    ax2.set_ylabel('Distance (miles)', color=color_miles, fontsize=12)
    line2 = ax2.plot(system_numbers, sorted_distances_miles, color='black', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_miles)
    
    # Set right y-axis to also start from 0
    ax2.set_ylim(0, sorted_distances_miles.max() * 1.05)
    
    plt.tight_layout()
    plt.savefig('Output Figure/Water_Systems_Nearest_Neighbor_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: Output Figure/Water_Systems_Nearest_Neighbor_Analysis.png")
    
    # Create consolidation sensitivity plot
    create_consolidation_sensitivity_plot()
    
    return nearest_distances_km, nearest_distances_miles

def create_consolidation_sensitivity_plot():
    """Create consolidation sensitivity analysis plot"""
    try:
        print("\nLoading consolidation sensitivity analysis results...")
        sensitivity_df = pd.read_csv('Output Data/Consolidation_Sensitivity_Analysis.csv')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use fresh, soft colors
        colors = {
            'joint': 'blue',
            'balanced': 'green', 
            'direct': 'red',
            'total': 'purple'
        }
        
        # Plot each consolidation type
        ax.plot(sensitivity_df['Distance_km'], sensitivity_df['Joint_Merger'], '-', 
                color=colors['joint'], linewidth=2.5)
        ax.plot(sensitivity_df['Distance_km'], sensitivity_df['Balanced_Merger'], '-', 
                color=colors['balanced'], linewidth=2.5)
        ax.plot(sensitivity_df['Distance_km'], sensitivity_df['Direct_Acquisition'], '-', 
                color=colors['direct'], linewidth=2.5)
        ax.plot(sensitivity_df['Distance_km'], sensitivity_df['Total_Consolidation'], '-', 
                color=colors['total'], linewidth=3)
        
        # Add text labels directly on the lines
        mid_x = sensitivity_df['Distance_km'].iloc[len(sensitivity_df)//2]
        
        # Find y-values at the middle x position for labeling
        mid_idx = len(sensitivity_df) // 2
        joint_y = sensitivity_df['Joint_Merger'].iloc[mid_idx]
        balanced_y = sensitivity_df['Balanced_Merger'].iloc[mid_idx]
        direct_y = sensitivity_df['Direct_Acquisition'].iloc[mid_idx]
        total_y = sensitivity_df['Total_Consolidation'].iloc[mid_idx]
        
        # Add text labels
        ax.text(mid_x, joint_y + 50, 'Joint Merger', fontsize=11, ha='center', 
                color=colors['joint'], fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.text(mid_x + 10, balanced_y + 100, 'Balanced Merger', fontsize=11, ha='center', 
                color=colors['balanced'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.text(mid_x, direct_y - 100, 'Direct Acquisition', fontsize=11, ha='center', 
                color=colors['direct'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.text(mid_x, total_y + 50, 'Total Consolidation', fontsize=11, ha='center', 
                color=colors['total'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Customize plot
        ax.set_xlabel('Distance Threshold (km)', fontsize=12)
        ax.set_ylabel('Number of Systems', fontsize=12)
        ax.grid(True, alpha=0.2)  # Lighter grid for fresh look
        
        # Set reasonable axis limits
        ax.set_xlim(0, sensitivity_df['Distance_km'].max())
        ax.set_ylim(0, max(sensitivity_df['Total_Consolidation'].max(), 
                          sensitivity_df['Joint_Merger'].max(),
                          sensitivity_df['Balanced_Merger'].max(),
                          sensitivity_df['Direct_Acquisition'].max()) * 1.05)
        
        plt.tight_layout()
        plt.savefig('Output Figure/Consolidation_Sensitivity_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Consolidation sensitivity plot saved to: Output Figure/Consolidation_Sensitivity_Analysis.png")
        
    except FileNotFoundError:
        print("Consolidation sensitivity analysis results not found.")
        print("Please run 'consolidation_sensitivity_analysis.py' first to generate the results.")

if __name__ == "__main__":
    results = water_systems_nearest_neighbor_analysis()