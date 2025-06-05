# Script for sensitivity analysis of water system consolidation thresholds
# Creates a simple table showing consolidation types across different distance thresholds

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all required data and ensure proper index alignment"""
    try:
        print("Loading CWS data...")
        cws_ca = pd.read_csv('Output Data/CWS_CA.csv')
        
        print("Loading distance matrix...")
        data = np.load('Output Data/PWSID_Distance_Matrix_km.npz')
        distance_matrix_km = data['distance_matrix']
        pwsids = data['pwsids']
        
        print("Loading clustering lookup table...")
        lookup_df = pd.read_csv('Output Data/Clustering_Lookup_Table.csv')
        
        # Align CWS data with distance matrix order
        print("Aligning CWS data with distance matrix...")
        cws_aligned = []
        missing_count = 0
        
        for pwsid in pwsids:
            matching_row = cws_ca[cws_ca['PWS.ID'] == pwsid]
            if len(matching_row) > 0:
                cws_aligned.append(matching_row.iloc[0])
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} PWS IDs in distance matrix not found in CWS data")
        
        cws_ca_aligned = pd.DataFrame(cws_aligned).reset_index(drop=True)
        print(f"Successfully aligned {len(cws_ca_aligned)} systems")
        
        print("All data loaded and aligned successfully!")
        return cws_ca_aligned, distance_matrix_km, lookup_df
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def find_optimal_k_for_threshold(lookup_df, threshold_km):
    """Find optimal k for a given distance threshold using lookup table"""
    valid_k = lookup_df[lookup_df['max_intra_cluster_distance_km'] <= threshold_km]
    return valid_k['k'].min() if len(valid_k) > 0 else len(lookup_df)

def analyze_consolidation_at_threshold(cws_data, distance_matrix, lookup_df, threshold_km):
    """Analyze consolidation for a specific distance threshold"""
    
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
        result = cws_jm
    
    # Count each type
    if 'Consolidation_Type' in result.columns:
        counts = result['Consolidation_Type'].value_counts()
        joint = counts.get('Joint_Merger', 0)
        balanced = counts.get('Balanced_Merger', 0)
        direct = counts.get('Direct_Acquisition', 0)
        none = counts.get('No_Consolidation', 0)
    else:
        joint = balanced = direct = 0
        none = len(result) if len(result) > 0 else len(cws_data)
    
    return {
        'Distance_km': threshold_km,
        'Joint_Merger': joint,
        'Balanced_Merger': balanced,
        'Direct_Acquisition': direct,
        'No_Consolidation': none,
        'Total_Consolidation': joint + balanced + direct
    }

def main():
    """Main function for sensitivity analysis"""
    
    # Load data with proper alignment
    cws_ca, distance_matrix, lookup_df = load_data()
    if cws_ca is None:
        return
    
    print(f"\nLoaded and aligned data for {len(cws_ca)} water systems")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Verify perfect alignment
    if len(cws_ca) == distance_matrix.shape[0] == distance_matrix.shape[1]:
        print("Perfect alignment verified")
    else:
        print("Dimension mismatch detected!")
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
    
    print(f"Running sensitivity analysis from 0 km to {max_meaningful_distance} km...")
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
    
    # Save results
    results_df.to_csv('Output Data/Consolidation_Sensitivity_Analysis.csv', index=False)
    print(f"\nResults saved to: Output Data/Consolidation_Sensitivity_Analysis.csv")
    print(f"Analysis completed successfully!")
    
    return results_df

if __name__ == "__main__":
    results = main()