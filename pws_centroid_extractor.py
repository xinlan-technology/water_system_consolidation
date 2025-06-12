# Water Service Areas Processing Script
# Converts water system service area shapefiles to centroids with coordinates

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

def process_water_service_areas(shapefile_path, output_csv_path):
    """
    Process water system service area shapefiles to extract centroid coordinates.
    
    Args:
        shapefile_path (str): Path to the input shapefile
        output_csv_path (str): Path for the output CSV file
    """
    
    # Load the shapefile containing water service areas
    print("Loading shapefile...")
    service = gpd.read_file(shapefile_path)
    print(f"Initial data shape: {service.shape}")
    
    # Filter out invalid geometries (keep only 2D polygons)
    # Remove rows where geometry is None or not a proper 2D polygon
    print("Filtering valid geometries...")
    service = service[service.geometry.notna()]
    service = service[service.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
    print(f"After geometry filtering: {service.shape}")
    
    # Select only required columns: pwsid and geometry
    service = service[['pwsid', 'geometry']].copy()
    
    # Convert pwsid to string format for consistency
    service['pwsid'] = service['pwsid'].astype(str)
    
    # Handle duplicate pwsid values by selecting the largest area
    print("Processing duplicate pwsid entries...")
    
    # Check for duplicates
    duplicate_count = service[service.duplicated(subset=['pwsid'], keep=False)].shape[0]
    if duplicate_count > 0:
        print(f"Found {duplicate_count} records with duplicate pwsids")
        
        # For each pwsid, keep only the geometry with the largest area
        print("Selecting largest area for each duplicate pwsid...")
        
        # Calculate area for each geometry (in projected coordinates for accuracy)
        service_with_area = service.to_crs('EPSG:3311')
        service['area'] = service_with_area.geometry.area
        
        # Keep the row with maximum area for each pwsid
        service = service.loc[service.groupby('pwsid')['area'].idxmax()].reset_index(drop=True)
        
        # Remove the temporary area column
        service = service.drop('area', axis=1)
        
        print(f"After selecting largest areas: {service.shape}")
        
        # Log some statistics about the selection process
        unique_pwsids = service['pwsid'].nunique()
        print(f"Final dataset: {len(service)} records for {unique_pwsids} unique water systems")
    
    else:
        print("No duplicate pwsids found")
    print(f"After removing duplicates: {service.shape}")
    
    # Calculate centroids of service areas
    print("Calculating centroids...")
    
    # Transform to projected coordinate system for accurate centroid calculation
    # EPSG:3311 is California Albers
    service_projected = service.to_crs('EPSG:3311')
    
    # Calculate centroids in projected coordinates
    centroids_projected = service_projected.geometry.centroid
    
    # Transform centroids back to geographic coordinates (WGS84)
    centroids_geo = centroids_projected.to_crs('EPSG:4326')
    
    # Extract longitude and latitude coordinates
    coordinates = []
    for point in centroids_geo:
        coordinates.append({
            'longitude': point.x,
            'latitude': point.y
        })
    
    # Create final dataframe with coordinates and pwsid
    center_df = pd.DataFrame(coordinates)
    center_df['pws.id'] = service['pwsid'].values
    
    # Reorder columns for better readability
    center_df = center_df[['pws.id', 'longitude', 'latitude']]
    
    # Export to CSV
    print(f"Exporting {len(center_df)} records to CSV...")
    center_df.to_csv(output_csv_path, index=False)
    
    print(f"Processing complete! Output saved to: {output_csv_path}")
    return center_df

# Main execution
if __name__ == "__main__":
    # Define file paths
    shapefile_path = "Input Data/cws_shape_file/service_areas.shp"
    
    # Set output directory and file path
    output_csv_path = "Output Data/location.csv"
    
    # Process the data
    try:
        result_df = process_water_service_areas(shapefile_path, output_csv_path)
        print("\nSample of output data:")
        print(result_df.head())
        print(f"\nTotal records processed: {len(result_df)}")
        
    except FileNotFoundError:
        print(f"Error: Shapefile not found at {shapefile_path}")
    except Exception as e:
        print(f"Error processing data: {str(e)}")