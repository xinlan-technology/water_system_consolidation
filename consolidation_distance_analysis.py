# Water system consolidation visualization with characteristic analysis
# Creates plots for total consolidation and specific characteristics (health violations, monitoring violations, population decline, SAFER status)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def water_systems_nearest_neighbor_analysis():
    """Calculate and visualize nearest neighbor distances for California water systems"""
    
    print("Loading data...")
    try:
        cws_ca = pd.read_csv('Output Data/CWS_CA.csv')
        data = np.load('Output Data/PWSID_Distance_Matrix_km.npz')
        distance_matrix_km = data['distance_matrix']
        print(f"Loaded {len(cws_ca)} water systems")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate nearest neighbor distances
    print("Calculating nearest neighbor distances...")
    dist_matrix = distance_matrix_km.copy()
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_distances_km = np.min(dist_matrix, axis=1)
    nearest_distances_miles = nearest_distances_km * 0.621371
    
    # Statistics
    print(f"\nDistance Statistics:")
    print(f"  Minimum:  {nearest_distances_km.min():.2f} km ({nearest_distances_miles.min():.2f} miles)")
    print(f"  Maximum:  {nearest_distances_km.max():.2f} km ({nearest_distances_miles.max():.2f} miles)")
    print(f"  Median:   {np.median(nearest_distances_km):.2f} km ({np.median(nearest_distances_miles):.2f} miles)")
    print(f"  Mean:     {nearest_distances_km.mean():.2f} km ({nearest_distances_miles.mean():.2f} miles)")
    
    # Plot nearest neighbor distances
    fig, ax1 = plt.subplots(figsize=(12, 8))
    sorted_distances_km = np.sort(nearest_distances_km)
    sorted_distances_miles = np.sort(nearest_distances_miles)
    system_numbers = np.arange(1, len(sorted_distances_km) + 1)
    
    ax1.plot(system_numbers, sorted_distances_km, 'black', linewidth=1.5)
    ax1.set_xlabel('System Number (Sorted by Distance)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Distance (km)', color='tab:blue', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14, labelcolor='black', width=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(system_numbers, sorted_distances_miles, 'black', linewidth=1.5)
    ax2.set_ylabel('Distance (miles)', color='tab:red', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', which='major', labelsize=14, labelcolor='tab:red', width=2)
    
    plt.tight_layout()
    plt.savefig('Output Figure/Water_Systems_Nearest_Neighbor_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Nearest neighbor plot saved.")
    
    return nearest_distances_km, nearest_distances_miles

def create_consolidation_plot(df, plot_type, title, filename):
    """Create a single consolidation plot"""
    
    # Color palette
    colors = {
        'joint': '#0173B2',      # Blue
        'balanced': '#DE8F05',   # Orange  
        'direct': '#029E73',     # Teal/Green
        'total': '#CC78BC'       # Pink/Purple
    }
    
    # Define column mappings for different plot types
    plot_configs = {
        'total': {
            'columns': ['Joint_Merger_Total', 'Balanced_Merger_Total', 'Direct_Acquisition_Total', 'Total_Consolidation'],
            'labels': ['Joint Merger', 'Balanced Merger', 'Direct Acquisition', 'Total Consolidation'],
            'colors': [colors['joint'], colors['balanced'], colors['direct'], colors['total']]
        },
        'health': {
            'columns': ['Joint_Merger_Health_Violation', 'Balanced_Merger_Health_Violation', 
                       'Direct_Acquisition_Health_Violation'],
            'labels': ['Joint Merger', 'Balanced Merger', 'Direct Acquisition', 'Total Consolidation'],
            'colors': [colors['joint'], colors['balanced'], colors['direct'], colors['total']]
        },
        'monitoring': {
            'columns': ['Joint_Merger_Monitoring_Violation', 'Balanced_Merger_Monitoring_Violation',
                       'Direct_Acquisition_Monitoring_Violation'],
            'labels': ['Joint Merger', 'Balanced Merger', 'Direct Acquisition', 'Total Consolidation'],
            'colors': [colors['joint'], colors['balanced'], colors['direct'], colors['total']]
        },
        'population': {
            'columns': ['Joint_Merger_Decreasing_Population', 'Balanced_Merger_Decreasing_Population',
                       'Direct_Acquisition_Decreasing_Population'],
            'labels': ['Joint Merger', 'Balanced Merger', 'Direct Acquisition', 'Total Consolidation'],
            'colors': [colors['joint'], colors['balanced'], colors['direct'], colors['total']]
        },
        'safer': {
            'columns': ['Joint_Merger_SAFER_Failing', 'Balanced_Merger_SAFER_Failing',
                       'Direct_Acquisition_SAFER_Failing'],
            'labels': ['Joint Merger', 'Balanced Merger', 'Direct Acquisition', 'Total Consolidation'],
            'colors': [colors['joint'], colors['balanced'], colors['direct'], colors['total']]
        }
    }
    
    config = plot_configs[plot_type]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot lines - for total plot, plot all 4 lines; for characteristic plots, only plot first 3
    num_lines = len(config['columns']) if plot_type == 'total' else 3
    
    # Plot lines
    for i in range(num_lines):
        col = config['columns'][i]
        label = config['labels'][i]
        color = config['colors'][i]
        
        if col in df.columns:
            linewidth = 3 if 'Total' in label else 2.5
            ax.plot(df['Distance_km'], df[col], '-', color=color, linewidth=linewidth, label=label)
    
    # For characteristic plots, add the "Total Consolidation" line (sum of three consolidation types)
    if plot_type != 'total':
        total_consolidation_characteristic = (df[config['columns'][0]] + 
                                            df[config['columns'][1]] + 
                                            df[config['columns'][2]])
        ax.plot(df['Distance_km'], total_consolidation_characteristic, '-', 
               color=config['colors'][3], linewidth=3, label=config['labels'][3])
    
    # Customize plot
    ax.set_xlabel('Distance Threshold (km)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Systems', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, df['Distance_km'].max())
    
    # Set y-axis limit
    if plot_type == 'total':
        max_val = max([df[col].max() for col in config['columns'] if col in df.columns])
    else:
        # For characteristic plots, include the calculated total in max calculation
        individual_maxes = [df[col].max() for col in config['columns']]
        total_max = (df[config['columns'][0]] + df[config['columns'][1]] + df[config['columns'][2]]).max()
        max_val = max(individual_maxes + [total_max])
    
    ax.set_ylim(0, max_val * 1.05)
    
    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=2)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.95), frameon=False, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'Output Figure/{filename}', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {filename}")

def create_all_consolidation_plots():
    """Create all consolidation sensitivity plots"""
    try:
        print("\nLoading consolidation sensitivity analysis results...")
        df = pd.read_csv('Output Data/Consolidation_Sensitivity_Analysis.csv')
        
        # Plot configurations
        plot_configs = [
            {
                'type': 'total',
                'title': 'Water System Consolidation by Distance Threshold',
                'filename': 'Consolidation_Total_Analysis.png'
            },
            {
                'type': 'health',
                'title': 'Health Violations in Consolidated Systems',
                'filename': 'Consolidation_Health_Violation_Analysis.png'
            },
            {
                'type': 'monitoring',
                'title': 'Monitoring Violations in Consolidated Systems',
                'filename': 'Consolidation_Monitoring_Violation_Analysis.png'
            },
            {
                'type': 'population',
                'title': 'Decreasing Population in Consolidated Systems',
                'filename': 'Consolidation_Decreasing_Population_Analysis.png'
            },
            {
                'type': 'safer',
                'title': 'SAFER Failing Systems in Consolidation',
                'filename': 'Consolidation_SAFER_Failing_Analysis.png'
            }
        ]
        
        # Create all plots
        for config in plot_configs:
            create_consolidation_plot(df, config['type'], config['title'], config['filename'])
        
        print(f"\nGenerated 5 consolidation analysis plots successfully!")
        
    except FileNotFoundError:
        print("Consolidation sensitivity analysis results not found.")

if __name__ == "__main__":
    
    # Calculate nearest neighbor distances
    results = water_systems_nearest_neighbor_analysis()
        
    # Create all consolidation plots
    create_all_consolidation_plots()