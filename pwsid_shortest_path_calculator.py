# California Water Systems Shortest Path Distance Calculator Script
# This script computes pairwise shortest path distances between community water systems
# using PWSID as system identifier

import osmnx as ox
import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import BallTree
import os

## SECTION 1: Load California Road Network
print("Loading California road network...")
G_nx = nx.read_graphml("Output Data/CA_RoadNetwork.graphml")
print(f"Number of nodes: {len(G_nx.nodes):,}")
print(f"Number of edges: {len(G_nx.edges):,}")

## SECTION 2: Convert NetworkX Graph to iGraph
print("Converting to iGraph for faster computation...")

# Initialize iGraph
G_ig = ig.Graph(directed=False)

# Create node mappings
nx_nodes = list(G_nx.nodes)
nx_node_to_idx = {node: idx for idx, node in enumerate(nx_nodes)}

# Add vertices
G_ig.add_vertices(len(nx_nodes))

# Build edges with weights
edges = []
weights = []

for u, v, data in tqdm(G_nx.edges(data=True), desc="Converting edges"):
    length = data.get("length")
    if length is not None and length > 0:
        idx_u = nx_node_to_idx[u]
        idx_v = nx_node_to_idx[v]
        edges.append((idx_u, idx_v))
        weights.append(length)

# Add edges to iGraph
G_ig.add_edges(edges)
G_ig.es["weight"] = weights

print(f"iGraph vertices: {G_ig.vcount():,}")
print(f"iGraph edges: {G_ig.ecount():,}")

## SECTION 3: Load Water Systems and Match to Road Network
print("Loading water system locations...")
CWS_Location = pd.read_csv("Output Data/CWS_CA.csv")
print(f"Loaded {len(CWS_Location)} water systems")

# Extract coordinates
latitudes = CWS_Location["Latitude"].values
longitudes = CWS_Location["Longitude"].values

# Extract road network coordinates
node_ids = list(G_nx.nodes)
node_coords = np.array([(G_nx.nodes[node]["y"], G_nx.nodes[node]["x"]) for node in node_ids])

# Convert to radians for haversine distance
system_coords_rad = np.radians(np.column_stack((latitudes, longitudes)))
node_coords_rad = np.radians(node_coords)

# Find nearest road nodes
print("Matching systems to nearest road nodes...")
tree = BallTree(node_coords_rad, metric='haversine')
dist_rad, indices = tree.query(system_coords_rad, k=1)

# Get iGraph node indices
nearest_node_ids = [node_ids[i[0]] for i in indices]
igraph_node_indices = [nx_node_to_idx[node] for node in nearest_node_ids]

CWS_Location["igraph_node"] = igraph_node_indices

## SECTION 4: Compute Distance Matrix Between PWSIDs
print("Computing shortest path distances between PWSIDs...")

# Get unique nodes to optimize computation
all_nodes = np.array(CWS_Location["igraph_node"].tolist())
unique_nodes, inverse_indices = np.unique(all_nodes, return_inverse=True)

print(f"Total systems: {len(all_nodes)}")
print(f"Unique road nodes: {len(unique_nodes)}")

# Compute distances for unique nodes only
distance_unique = np.array(
    G_ig.distances(source=unique_nodes.tolist(), 
                  target=unique_nodes.tolist(), 
                  weights="weight")
)

# Expand to full distance matrix
distance_matrix_meters = distance_unique[inverse_indices[:, None], inverse_indices]

# Convert to kilometers
distance_matrix_km = distance_matrix_meters / 1000.0

print(f"Distance matrix shape: {distance_matrix_km.shape}")
print(f"Average distance: {np.mean(distance_matrix_km[distance_matrix_km > 0]):.1f} km")

## SECTION 5: Create PWSID Distance DataFrame and Save
print("Creating PWSID distance matrix...")

# Get PWSID list
pwsids = CWS_Location["PWSID"].tolist()

# Create DataFrame with PWSID as index and columns
distance_df = pd.DataFrame(
    distance_matrix_km,
    index=pwsids,
    columns=pwsids
)

# Save the distance matrix
output_file = "Output Data/PWSID_Distance_Matrix_km.csv"
distance_df.to_csv(output_file)

print(f"PWSID distance matrix saved to: {output_file}")
print(f"Matrix dimensions: {distance_df.shape[0]} systems × {distance_df.shape[1]} systems")
print("Computation completed successfully!")