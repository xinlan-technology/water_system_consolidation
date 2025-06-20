# Script to precompute clustering lookup table for sensitivity analysis
# Calculates optimal k and maximum intra-cluster distances for all possible cluster numbers

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, cut_tree
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load CWS data and distance matrix with proper alignment"""
    try:
        print("Loading CWS data...")
        cws_ca = pd.read_csv('Output Data/CWS_CA.csv')
        print(f"Loaded {len(cws_ca)} water systems")
        
        print("Loading distance matrix...")
        data = np.load('Output Data/PWSID_Distance_Matrix_km.npz')
        distance_matrix_km = data['distance_matrix']
        pwsids = data['pwsids']
        print(f"Loaded distance matrix: {distance_matrix_km.shape[0]} × {distance_matrix_km.shape[1]} systems")
        
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
        
        return cws_ca_aligned, distance_matrix_km, pwsids
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def get_max_distance_in_cluster(cluster_indices, distance_matrix):
    """Calculate maximum distance within a cluster"""
    if len(cluster_indices) <= 1:
        return 0
    
    max_dist = 0
    for i in range(len(cluster_indices)):
        for j in range(i+1, len(cluster_indices)):
            idx1, idx2 = cluster_indices[i], cluster_indices[j]
            dist = distance_matrix[idx1, idx2]
            max_dist = max(max_dist, dist)
    
    return max_dist

def create_clustering_lookup_table(cws_data, distance_matrix):
    """Create lookup table for k vs max intra-cluster distance"""
    
    # Prepare features for clustering
    features = ['Longitude', 'Latitude', 'Population.2021', 'Health.violation', 
               'Monitoring.and.reporting.violation']
    
    print(f"\nUsing {len(cws_data)} systems for clustering")
    
    # Standardize data
    print("Standardizing features...")
    scaler = StandardScaler()
    cws_scaled = scaler.fit_transform(cws_data[features])
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    linkage_matrix = linkage(cws_scaled, method='complete')
    print("Clustering completed!")
    
    # Create lookup table
    print("Creating lookup table...")
    n_systems = len(cws_data)
    lookup_table = []
    
    for k in range(1, n_systems + 1):
        if k % 100 == 0:
            print(f"Processing k = {k}/{n_systems}")
        
        # Get cluster labels for k clusters
        cluster_labels = cut_tree(linkage_matrix, n_clusters=k).flatten()
        
        # Calculate max distance for each cluster
        max_distances = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            max_dist = get_max_distance_in_cluster(cluster_indices, distance_matrix)
            max_distances.append(max_dist)
        
        # Record the maximum distance among all clusters
        overall_max_distance_km = max(max_distances) if max_distances else 0
        
        lookup_table.append({
            'k': k,
            'max_intra_cluster_distance_km': overall_max_distance_km
        })
    
    return pd.DataFrame(lookup_table)

def main():
    """Main function to create clustering lookup table"""
    
    # Load data
    cws_data, distance_matrix, pwsids = load_data()
    if cws_data is None:
        return
    
    print(f"\nStarting clustering lookup table creation for {len(cws_data)} aligned systems...")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Verify perfect alignment
    if len(cws_data) == distance_matrix.shape[0] == distance_matrix.shape[1]:
        print("Perfect alignment: CWS data and distance matrix dimensions match")
    else:
        print("Dimension mismatch detected!")
        return
    
    # Create lookup table
    lookup_df = create_clustering_lookup_table(cws_data, distance_matrix)
    
    # Save lookup table
    lookup_filename = 'Output Data/Clustering_Lookup_Table.csv'
    lookup_df.to_csv(lookup_filename, index=False)
    print(f"\nLookup table saved to: {lookup_filename}")
    
    print(f"Lookup table creation completed successfully!")
    return lookup_df

if __name__ == "__main__":
    lookup_table = main()