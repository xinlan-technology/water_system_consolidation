# Consolidation Case Studies Analysis
# Performs comprehensive water system consolidation analysis for two key scenarios:
# 1. Physical Consolidation (1 mile threshold) - systems within walking/piping distance
# 2. Managerial Consolidation (10 km threshold) - systems suitable for shared management
# 
# Output: Enhanced CWS dataset with consolidation type classifications for both scenarios
# Consolidation types: Joint_Merger, Balanced_Merger, Direct_Acquisition, No_Consolidation

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
    """Find optimal k for a given distance threshold"""
    valid_k = lookup_df[lookup_df['max_intra_cluster_distance_km'] <= threshold_km]
    return valid_k['k'].min() if len(valid_k) > 0 else len(lookup_df)

def get_consolidation_type_with_clusters(cws_data, distance_matrix, lookup_df, threshold_km):
    """Get consolidation type and cluster information for each system at given threshold"""
    
    # Prepare features for clustering
    features = ['Longitude', 'Latitude', 'Population.2021', 'Health.violation', 
               'Monitoring.and.reporting.violation']
    
    # Initial clustering using all features
    optimal_k = find_optimal_k_for_threshold(lookup_df, threshold_km)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cws_data[features])
    linkage_matrix = linkage(features_scaled, method='complete')
    cluster_labels = cut_tree(linkage_matrix, n_clusters=optimal_k).flatten()
    
    # Initialize consolidation types and temporary cluster assignments
    consolidation_types = ['No_Consolidation'] * len(cws_data)
    temp_cluster_assignments = list(range(len(cws_data)))
    
    # Track which systems belong to which consolidation groups
    consolidation_groups = []
    
    # Step 1: Handle Joint Mergers (multi-system initial clusters)
    cluster_sizes = pd.Series(cluster_labels).value_counts()
    joint_merger_clusters = cluster_sizes[cluster_sizes > 1].index
    
    for cluster_id in joint_merger_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0].tolist()
        consolidation_groups.append(cluster_indices)
        for idx in cluster_indices:
            consolidation_types[idx] = 'Joint_Merger'
    
    # Step 2: Geographic clustering for single systems
    single_clusters = cluster_sizes[cluster_sizes == 1].index
    single_indices = []
    
    for cluster_id in single_clusters:
        idx = np.where(cluster_labels == cluster_id)[0][0]
        single_indices.append(idx)
    
    if len(single_indices) > 1:
        # Geographic distance clustering
        geo_distances = distance_matrix[np.ix_(single_indices, single_indices)] * 1000  # to meters
        geo_distances_condensed = squareform(geo_distances, checks=False)
        geo_linkage = linkage(geo_distances_condensed, method='complete')
        geo_clusters = cut_tree(geo_linkage, height=threshold_km * 1000).flatten()
        
        # Classify each geographic cluster
        for geo_cluster_id in np.unique(geo_clusters):
            cluster_member_indices = np.where(geo_clusters == geo_cluster_id)[0]
            
            if len(cluster_member_indices) > 1:
                # Multiple systems in geographic cluster
                original_indices = [single_indices[i] for i in cluster_member_indices]
                populations = cws_data.iloc[original_indices]['Population.2021'].values
                
                # Sort populations to find largest and second largest
                sorted_pops = sorted(populations, reverse=True)
                largest = sorted_pops[0]
                second_largest = sorted_pops[1] if len(sorted_pops) > 1 else sorted_pops[0]
                
                # Calculate ratio and determine consolidation type
                ratio = second_largest / largest if largest > 0 else 1.0
                cons_type = 'Direct_Acquisition' if ratio <= 0.1 else 'Balanced_Merger'
                
                # Add this as a consolidation group
                consolidation_groups.append(original_indices)
                for orig_idx in original_indices:
                    consolidation_types[orig_idx] = cons_type
            else:
                # Single system - will be handled later as individual cluster
                pass
    
    # Step 3: Assign continuous cluster numbers
    cluster_assignments = [0] * len(cws_data)
    next_cluster_id = 0
    
    # First, assign cluster IDs to consolidation groups
    for group in consolidation_groups:
        for idx in group:
            cluster_assignments[idx] = next_cluster_id
        next_cluster_id += 1
    
    # Then, assign cluster IDs to individual systems (No_Consolidation)
    for i in range(len(cws_data)):
        if consolidation_types[i] == 'No_Consolidation':
            cluster_assignments[i] = next_cluster_id
            next_cluster_id += 1
    
    return consolidation_types, cluster_assignments

def main():
    """Main function"""
    
    # Load data with proper alignment
    cws_ca, distance_matrix, lookup_df = load_data()
    if cws_ca is None:
        return
    
    print(f"Processing {len(cws_ca)} aligned water systems...")
    
    # Verify perfect alignment
    if len(cws_ca) == distance_matrix.shape[0] == distance_matrix.shape[1]:
        print("Perfect alignment verified")
    else:
        print("Dimension mismatch detected!")
        return
    
    # Define thresholds
    physical_threshold_km = 1.609344  # 1 mile
    managerial_threshold_km = 10.0    # 10 km
    
    print(f"Analyzing Physical Consolidation (1 mile = {physical_threshold_km:.3f} km)...")
    physical_types, physical_clusters = get_consolidation_type_with_clusters(
        cws_ca, distance_matrix, lookup_df, physical_threshold_km)
    
    print(f"Analyzing Managerial Consolidation ({managerial_threshold_km} km)...")
    managerial_types, managerial_clusters = get_consolidation_type_with_clusters(
        cws_ca, distance_matrix, lookup_df, managerial_threshold_km)
    
    # Add consolidation type and cluster columns to original CWS data
    result_df = cws_ca.copy()
    result_df['Physical_Consolidation_Type'] = physical_types
    result_df['Physical_Cluster_Number'] = physical_clusters
    result_df['Managerial_Consolidation_Type'] = managerial_types
    result_df['Managerial_Cluster_Number'] = managerial_clusters
    
    # Save results
    output_filename = 'Output Data/CWS_CA_Case_Study_Results.csv'
    result_df.to_csv(output_filename, index=False)
    
    print(f"\nResults saved to: {output_filename}")
    
    print(f"\nAnalysis completed successfully!")
    return result_df

if __name__ == "__main__":
    results_df = main()