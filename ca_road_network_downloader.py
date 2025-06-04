# California Road Network Downloader Script
# Downloads California road network data from OpenStreetMap using OSMnx
# Saves data as GraphML format and verifies successful loading

import osmnx as ox
import networkx as nx
import os

def download_ca_road_network():
    """
    Download California road network using OSMnx and save as GraphML
    """
    print("Downloading California road network from OpenStreetMap...")
    
    # Download road network for California
    try:
        G = ox.graph_from_place("California, USA", network_type='drive')
        print("Download completed successfully!")
        
        # Save to GraphML format in Output Data folder
        output_path = os.path.join("Output Data", "CA_RoadNetwork.graphml")
        ox.save_graphml(G, output_path)
        print("Data saved to:", output_path)
        
        return G, output_path
        
    except Exception as e:
        print("Error downloading California data:", e)
        return None, None

def verify_graph_data(G, filepath):
    """
    Load and verify the downloaded graph data
    """
    print("\nLoading graph data from:", filepath)
    
    # Load the saved GraphML file
    G_loaded = nx.read_graphml(filepath)
    
    # Print basic information to confirm the data is loaded correctly
    print("Number of nodes:", len(G_loaded.nodes))
    print("Number of edges:", len(G_loaded.edges))
    
    print("Graph data loaded and verified successfully!")
    return G_loaded

def main():
    """
    Main function to download and verify CA road network
    """
    print("California Road Network Downloader")
    print("=" * 35)
    
    # Download the road network
    G, filepath = download_ca_road_network()
    
    if G is not None and filepath is not None:
        # Verify the downloaded data
        G_loaded = verify_graph_data(G, filepath)
        print("\nSuccess! Road network data is ready in:", filepath)
    else:
        print("\nFailed to download road network data")

if __name__ == "__main__":
    main()