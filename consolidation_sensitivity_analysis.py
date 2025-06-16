# This script is used to analyze the consolidation of water systems across different distance thresholds
# Water system consolidation sensitivity analysis across distance thresholds
# Analyzes consolidation types and characteristics (health violations, monitoring violations, population decline, SAFER status) across different distance thresholds

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all required data"""
    try:
        print("Loading CWS data...")
        cws_ca = pd.read_csv('Output Data/CWS_CA.csv')
        
        print("Loading distance matrix...")
        data = np.load('Output Data/PWSID_Distance_Matrix_km.npz')
        distance_matrix_km = data['distance_matrix']
        
        print("Loading clustering lookup table...")
        lookup_df = pd.read_csv('Output Data/Clustering_Lookup_Table.csv')
        
        print(f"Loaded {len(cws_ca)} water systems")
        print(f"Distance matrix shape: {distance_matrix_km.shape}")
        print("All data loaded successfully!")
        
        return cws_ca, distance_matrix_km, lookup_df
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def find_optimal_k_for_threshold(lookup_df, threshold_km):
    """Find optimal k for a given distance threshold using lookup table"""
    valid_k = lookup_df[lookup_df['max_intra_cluster_distance_km'] <= threshold_km]
    return valid_k['k'].min() if len(valid_k) > 0 else len(lookup_df)

def count_system_characteristics(df):
    """Count systems with specific characteristics"""
    if len(df) == 0:
        return {
            'health_violation': 0,
            'monitoring_violation': 0,
            'decreasing_population': 0,
            'safer_failing': 0
        }
    
    # Count systems with health violations (any value > 0 means violation)
    health_violation = 0
    if 'Health.violation' in df.columns:
        health_violation = int((df['Health.violation'] > 0).sum())
    
    # Count systems with monitoring violations (any value > 0 means violation)
    monitoring_violation = 0
    if 'Monitoring.and.reporting.violation' in df.columns:
        monitoring_violation = int((df['Monitoring.and.reporting.violation'] > 0).sum())
    
    # Count systems with decreasing population (negative Population.Change)
    decreasing_population = 0
    if 'Population.Change' in df.columns:
        decreasing_population = int((df['Population.Change'] < 0).sum())
    
    # Count systems with SAFER.STATUS = Failing
    safer_failing = 0
    if 'SAFER.STATUS' in df.columns:
        safer_failing = int((df['SAFER.STATUS'] == 'Failing').sum())
    
    return {
        'health_violation': health_violation,
        'monitoring_violation': monitoring_violation,
        'decreasing_population': decreasing_population,
        'safer_failing': safer_failing
    }

def analyze_consolidation_at_threshold(cws_data, distance_matrix, lookup_df, threshold_km):
    """Analyze consolidation for a specific distance threshold with detailed characteristics"""
    
    distance_threshold_meters = threshold_km * 1000
    
    # Prepare features for clustering
    features = ['Longitude', 'Latitude', 'Population.2021', 'Health.violation', 
               'Monitoring.and.reporting.violation']
    
    # Step 1: Initial clustering using all features
    optimal_k = find_optimal_k_for_threshold(lookup_df, threshold_km)
    scaler = StandardScaler()
    cws_scaled = scaler.fit_transform(cws_data[features])
    linkage_matrix = linkage(cws_scaled, method='complete')
    cluster_labels = cut_tree(linkage_matrix, n_clusters=optimal_k).flatten()
    
    cws_work = cws_data.copy()
    cws_work['Cluster'] = cluster_labels
    
    # Step 2: Identify Joint Mergers
    cluster_sizes = cws_work.groupby('Cluster').size().reset_index(name='Size')
    joint_merger_clusters = cluster_sizes[cluster_sizes['Size'] > 1]['Cluster'].values
    cws_jm = cws_work[cws_work['Cluster'].isin(joint_merger_clusters)].copy()
    cws_jm['Consolidation_Type'] = 'Joint_Merger'
    
    # Step 3: Geographic clustering for single systems
    single_clusters = cluster_sizes[cluster_sizes['Size'] == 1]['Cluster'].values
    cws_single = cws_work[cws_work['Cluster'].isin(single_clusters)].copy()
    
    consolidation_results = []
    
    if len(cws_single) > 0:
        # Use clean indices for distance matrix access
        single_indices = cws_single.index.values
        
        # Extract distance submatrix for single systems
        geo_distances = distance_matrix[np.ix_(single_indices, single_indices)] * 1000
        geo_distances_condensed = squareform(geo_distances, checks=False)
        geo_linkage = linkage(geo_distances_condensed, method='complete')
        geo_clusters = cut_tree(geo_linkage, height=distance_threshold_meters).flatten()
        cws_single = cws_single.copy()
        cws_single['GeoCluster'] = geo_clusters
        
        # Classify each geographic cluster
        for geo_cluster_id in cws_single['GeoCluster'].unique():
            cluster_data = cws_single[cws_single['GeoCluster'] == geo_cluster_id].copy()
            
            if len(cluster_data) == 1:
                cluster_data['Consolidation_Type'] = 'No_Consolidation'
            else:
                # Sort by population
                cluster_data = cluster_data.sort_values('Population.2021', ascending=False)
                populations = cluster_data['Population.2021'].values
                
                largest = populations[0]
                second_largest = populations[1] if len(populations) > 1 else populations[0]
                
                # Calculate ratio and determine consolidation type
                ratio = second_largest / largest if largest > 0 else 1.0
                
                if ratio <= 0.1:
                    cluster_data['Consolidation_Type'] = 'Direct_Acquisition'
                else:
                    cluster_data['Consolidation_Type'] = 'Balanced_Merger'
            
            consolidation_results.append(cluster_data)
    
    # Combine results
    if consolidation_results:
        cws_geographic = pd.concat(consolidation_results, ignore_index=True)
        result = pd.concat([cws_jm, cws_geographic], ignore_index=True)
    else:
        result = cws_jm.copy() if len(cws_jm) > 0 else pd.DataFrame()
    
    # Initialize result dictionary
    result_dict = {'Distance_km': threshold_km}
    
    # Count each consolidation type and their characteristics
    if 'Consolidation_Type' in result.columns and len(result) > 0:
        # Count total systems in each consolidation type
        counts = result['Consolidation_Type'].value_counts()
        
        # Joint Merger statistics
        joint_systems = result[result['Consolidation_Type'] == 'Joint_Merger']
        joint_chars = count_system_characteristics(joint_systems)
        result_dict.update({
            'Joint_Merger_Total': counts.get('Joint_Merger', 0),
            'Joint_Merger_Health_Violation': joint_chars['health_violation'],
            'Joint_Merger_Monitoring_Violation': joint_chars['monitoring_violation'],
            'Joint_Merger_Decreasing_Population': joint_chars['decreasing_population'],
            'Joint_Merger_SAFER_Failing': joint_chars['safer_failing']
        })
        
        # Balanced Merger statistics
        balanced_systems = result[result['Consolidation_Type'] == 'Balanced_Merger']
        balanced_chars = count_system_characteristics(balanced_systems)
        result_dict.update({
            'Balanced_Merger_Total': counts.get('Balanced_Merger', 0),
            'Balanced_Merger_Health_Violation': balanced_chars['health_violation'],
            'Balanced_Merger_Monitoring_Violation': balanced_chars['monitoring_violation'],
            'Balanced_Merger_Decreasing_Population': balanced_chars['decreasing_population'],
            'Balanced_Merger_SAFER_Failing': balanced_chars['safer_failing']
        })
        
        # Direct Acquisition statistics
        direct_systems = result[result['Consolidation_Type'] == 'Direct_Acquisition']
        direct_chars = count_system_characteristics(direct_systems)
        result_dict.update({
            'Direct_Acquisition_Total': counts.get('Direct_Acquisition', 0),
            'Direct_Acquisition_Health_Violation': direct_chars['health_violation'],
            'Direct_Acquisition_Monitoring_Violation': direct_chars['monitoring_violation'],
            'Direct_Acquisition_Decreasing_Population': direct_chars['decreasing_population'],
            'Direct_Acquisition_SAFER_Failing': direct_chars['safer_failing']
        })
        
        # No Consolidation statistics
        no_consol_systems = result[result['Consolidation_Type'] == 'No_Consolidation']
        no_consol_chars = count_system_characteristics(no_consol_systems)
        result_dict.update({
            'No_Consolidation_Total': counts.get('No_Consolidation', 0),
            'No_Consolidation_Health_Violation': no_consol_chars['health_violation'],
            'No_Consolidation_Monitoring_Violation': no_consol_chars['monitoring_violation'],
            'No_Consolidation_Decreasing_Population': no_consol_chars['decreasing_population'],
            'No_Consolidation_SAFER_Failing': no_consol_chars['safer_failing']
        })
        
        # Overall totals
        result_dict['Total_Systems'] = len(result)
        result_dict['Total_Consolidation'] = counts.get('Joint_Merger', 0) + counts.get('Balanced_Merger', 0) + counts.get('Direct_Acquisition', 0)
        
        # Overall characteristics across all systems
        all_chars = count_system_characteristics(result)
        result_dict.update({
            'Total_Health_Violation': all_chars['health_violation'],
            'Total_Monitoring_Violation': all_chars['monitoring_violation'],
            'Total_Decreasing_Population': all_chars['decreasing_population'],
            'Total_SAFER_Failing': all_chars['safer_failing']
        })
    
    return result_dict

def main():
    """Main function for enhanced sensitivity analysis"""
    
    # Load data
    cws_ca, distance_matrix, lookup_df = load_data()
    if cws_ca is None:
        return
    
    # Calculate nearest neighbor distances to determine the threshold range
    print("Calculating nearest neighbor distances for threshold range...")
    dist_matrix = distance_matrix.copy()
    np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-distances
    nearest_distances_km = np.min(dist_matrix, axis=1)
    max_meaningful_distance = int(np.ceil(nearest_distances_km.max()))
    
    print(f"Maximum nearest neighbor distance: {max_meaningful_distance} km")
    
    # Determine threshold range based on nearest neighbor distances
    threshold_range = list(range(0, max_meaningful_distance + 1))
    
    print(f"Running enhanced sensitivity analysis from 0 km to {max_meaningful_distance} km...")
    print(f"Number of thresholds to test: {len(threshold_range)}")
    
    # Run analysis
    import time
    start_time = time.time()
    
    results = []
    for i, threshold in enumerate(threshold_range):
        if i % 5 == 0:
            print(f"Progress: {i+1}/{len(threshold_range)} - Testing {threshold} km")
        
        result = analyze_consolidation_at_threshold(cws_ca, distance_matrix, lookup_df, threshold)
        results.append(result)
    
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds!")
    
    # Create results table
    results_df = pd.DataFrame(results)
    
    # Print all available columns to debug
    print(f"\nAll columns in results dataframe:")
    print(results_df.columns.tolist())
    
    # Save results
    results_df.to_csv('Output Data/Consolidation_Sensitivity_Analysis.csv', index=False)
    print(f"\nResults saved to: Output Data/Consolidation_Sensitivity_Analysis.csv")
    
    # Display summary statistics
    print(f"\nSummary of enhanced analysis:")
    print(f"Columns in output: {len(results_df.columns)}")
    print(f"Rows (distance thresholds): {len(results_df)}")
    print(f"Analysis completed successfully!")
    
    return results_df

if __name__ == "__main__":
    results = main()