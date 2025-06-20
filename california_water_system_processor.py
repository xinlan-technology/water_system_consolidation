# California Water System Data Processing Script
# Integrates population, geographic location, water source types, violation records, and HR2W data 
# for California public water systems to generate a unified analytical dataset

import pandas as pd
import numpy as np

def load_hr2w_data(input_folder):
    """
    Load and process HR2W (Human Right to Water) data
    
    Args:
        input_folder (str): Path to input data folder
        
    Returns:
        pandas.DataFrame or None: Processed HR2W data
    """
    hr2w_file = f'{input_folder}HR2W_2022_12.csv'
    
    try:
        # Read HR2W data with encoding handling
        try:
            hr2w = pd.read_csv(hr2w_file, encoding='utf-8')
        except UnicodeDecodeError:
            hr2w = pd.read_csv(hr2w_file, encoding='latin-1')
        
        # Select only the needed columns
        hr2w_list = hr2w[["PWSID", "SAFER STATUS"]].copy()
        
        # Rename to standardize (convert space to dot for consistency)
        hr2w_list.columns = ['PWS.ID', 'SAFER.STATUS']
        
        # Ensure PWS.ID is string for consistent merging
        hr2w_list['PWS.ID'] = hr2w_list['PWS.ID'].astype(str)
        
        print(f"Loaded {len(hr2w_list)} HR2W records")
        
        return hr2w_list
        
    except FileNotFoundError:
        print(f"Warning: Could not find HR2W file {hr2w_file}")
        return None
        
    except Exception as e:
        print(f"Error loading HR2W data: {str(e)}")
        return None

def process_california_water_systems():
    """
    Process California public water system data by integrating multiple data sources
    including population trends, violations, geographic information, and HR2W status.
    
    Returns:
        pandas.DataFrame: Processed dataset containing comprehensive water system information
    """
    
    # Read data files
    print("Loading data files...")

    # Input data files from Input Data folder
    input_folder = "Input Data/"
    
    # Read files with proper encoding and data type handling
    try:
        CWS_2016 = pd.read_csv(f'{input_folder}summary_16_4_ca.csv', encoding='utf-8')
    except UnicodeDecodeError:
        CWS_2016 = pd.read_csv(f'{input_folder}summary_16_4_ca.csv', encoding='latin-1')
    
    try:
        CWS_2021 = pd.read_csv(f'{input_folder}summary_21_4_ca.csv', encoding='utf-8')
    except UnicodeDecodeError:
        CWS_2021 = pd.read_csv(f'{input_folder}summary_21_4_ca.csv', encoding='latin-1')
    
    try:
        VIO_2021 = pd.read_csv(f'{input_folder}violation_21_4_ca.csv', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        VIO_2021 = pd.read_csv(f'{input_folder}violation_21_4_ca.csv', encoding='latin-1', low_memory=False)
    
    try:
        CWS_Detail = pd.read_csv(f'{input_folder}detail_21_4_ca.csv', encoding='utf-8')
    except UnicodeDecodeError:
        CWS_Detail = pd.read_csv(f'{input_folder}detail_21_4_ca.csv', encoding='latin-1')
    
    # Location file from Output Data folder (generated by pws_centroid_extractor.py)
    output_folder = "Output Data/"
    try:
        Location = pd.read_csv(f'{output_folder}location.csv', encoding='utf-8')
    except UnicodeDecodeError:
        Location = pd.read_csv(f'{output_folder}location.csv', encoding='latin-1')
    
    # Load HR2W data
    print("Loading HR2W data...")
    hr2w_data = load_hr2w_data(input_folder)
    
    # Create data frame with PWSID and primary source information
    print("Processing water system details...")
    
    # Extract PWSID, Primary Source, and Service Connections Count from the detail file
    CWS = CWS_Detail[["PWS ID", "Primary Source", "Service Connections Count"]].copy()
    
    # Standardize column names to match the rest of the code (convert spaces to dots)
    CWS.columns = ["PWS.ID", "Primary.Source", "Service.Connections.Count"]
    
    print(f"Initial CWS records: {len(CWS)}")
    
    # Standardize water source information
    water_source_mapping = {
        'Ground water purchased': 'Purchased water',
        'Surface water purchased': 'Purchased water',
        'Purchased ground water under influence of surface water source': 'Purchased water',
        'Groundwater under influence of surface water': 'Ground water',
        'Unknown Primary Source': 'Unknown source'
    }
    
    CWS['Primary.Source'] = CWS['Primary.Source'].replace(water_source_mapping)
    
    # Add location information to CWS data
    print("Adding location information...")

    # Location file has different column naming: pws.id (lowercase with dots)
    Location.columns = ["PWS.ID", "Longitude", "Latitude"]
    
    print(f"Location records: {len(Location)}")
    
    # Ensure PWS.ID columns have the same data type (convert both to string)
    CWS['PWS.ID'] = CWS['PWS.ID'].astype(str)
    Location['PWS.ID'] = Location['PWS.ID'].astype(str)
    
    # Merge CWS and Location data
    CWS = CWS.merge(Location, on="PWS.ID", how="inner")
    print(f"After location merge: {len(CWS)} records")
    
    # Select population information from CWS_2016 and CWS_2021
    print("Processing population data...")

    # Extract PWSID and Population Served Count from the summary files
    CWS_2016_pop = CWS_2016[["PWS ID", "Population Served Count"]].copy()
    CWS_2021_pop = CWS_2021[["PWS ID", "Population Served Count"]].copy()
    
    # Rename columns to standardized format (with dots)
    CWS_2016_pop.columns = ["PWS.ID", "Population.2016"]
    CWS_2021_pop.columns = ["PWS.ID", "Population.2021"]
    
    # Ensure PWS.ID columns are strings for consistent merging
    CWS_2016_pop['PWS.ID'] = CWS_2016_pop['PWS.ID'].astype(str)
    CWS_2021_pop['PWS.ID'] = CWS_2021_pop['PWS.ID'].astype(str)
    
    # Convert population columns to numeric, handling any commas or non-numeric values
    CWS_2016_pop['Population.2016'] = pd.to_numeric(
        CWS_2016_pop['Population.2016'].astype(str).str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    CWS_2021_pop['Population.2021'] = pd.to_numeric(
        CWS_2021_pop['Population.2021'].astype(str).str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    # Delete systems without population served in 2016
    print(f"Before filtering zero population: {len(CWS_2016_pop)} records")
    CWS_2016_pop = CWS_2016_pop[CWS_2016_pop['Population.2016'] != 0]
    print(f"After filtering zero population: {len(CWS_2016_pop)} records")
    
    # Combine population data (2016 and 2021)
    print(f"Population data in 2016 before merge: {len(CWS_2016_pop)} records")
    print(f"Population data in 2021 before merge: {len(CWS_2021_pop)} records")
    Pop = CWS_2016_pop.merge(CWS_2021_pop, on="PWS.ID", how="inner")
    print(f"Population data after merge: {len(Pop)} records")
    
    # Ensure both population columns are numeric before calculation
    Pop['Population.2016'] = pd.to_numeric(Pop['Population.2016'], errors='coerce').fillna(0)
    Pop['Population.2021'] = pd.to_numeric(Pop['Population.2021'], errors='coerce').fillna(0)
    
    # Calculate population change, avoiding division by zero
    Pop['Population.Change'] = np.where(
        Pop['Population.2016'] == 0, 
        0,  # If 2016 population is 0, set change to 0
        (Pop['Population.2021'] - Pop['Population.2016']) / Pop['Population.2016']
    )
    
    # Add population information to CWS data
    print(f"CWS records before population merge: {len(CWS)}")
    CWS = CWS.merge(Pop, on="PWS.ID", how="inner")
    print(f"After population merge: {len(CWS)} records")
    
    # Extract time information from violations
    print("Processing violation data...")
    VIO_2021_processed = VIO_2021.copy()

    # Extract year from violation first reported date
    VIO_2021_processed['Year'] = VIO_2021_processed['Violation First Reported Date'].str[-4:].astype(int)
    
    # Create violation categories
    VIO = VIO_2021_processed.copy()
    violation_mapping = {
        'MRDL': 'MCL',
        'MON': 'MR',
        'RPT': 'MR'
    }
    VIO['Violation Category Code'] = VIO['Violation Category Code'].replace(violation_mapping)
    
    # Select violation information from recent five years
    VIO = VIO[VIO['Year'] >= 2017]
    
    # Count violation frequency by violation category
    VIO_Fre = VIO.groupby(['PWS ID', 'Violation Category Code']).size().reset_index(name='Freq')
    VIO_Fre.rename(columns={'PWS ID': 'PWS.ID'}, inplace=True)
    
    # Ensure PWS.ID is string type for consistent merging
    VIO_Fre['PWS.ID'] = VIO_Fre['PWS.ID'].astype(str)
    
    # Create new violation category datasets
    MCL = VIO_Fre[VIO_Fre['Violation Category Code'] == "MCL"][['PWS.ID', 'Freq']].copy()
    TT = VIO_Fre[VIO_Fre['Violation Category Code'] == "TT"][['PWS.ID', 'Freq']].copy()
    MR = VIO_Fre[VIO_Fre['Violation Category Code'] == "MR"][['PWS.ID', 'Freq']].copy()
    
    # Add violation frequency information to CWS data
    print("Integrating violation information...")
    
    # MCL violations
    CWS = CWS.merge(MCL, on="PWS.ID", how="left")
    CWS['Freq'] = CWS['Freq'].fillna(0)
    CWS.rename(columns={'Freq': 'Maximum.contaminant.levels.violation'}, inplace=True)
    
    # TT violations
    CWS = CWS.merge(TT, on="PWS.ID", how="left")
    CWS['Freq'] = CWS['Freq'].fillna(0)
    CWS.rename(columns={'Freq': 'Treatment.technique.violation'}, inplace=True)
    
    # Calculate total health violations
    CWS['Health.violation'] = CWS['Maximum.contaminant.levels.violation'] + CWS['Treatment.technique.violation']
    
    # MR violations  
    CWS = CWS.merge(MR, on="PWS.ID", how="left")
    CWS['Freq'] = CWS['Freq'].fillna(0)
    CWS.rename(columns={'Freq': 'Monitoring.and.reporting.violation'}, inplace=True)
    
    # Add HR2W data to CWS
    print("Integrating HR2W data...")
    if hr2w_data is not None:
        # Merge with HR2W data (left join to keep all CWS records)
        before_hr2w_count = len(CWS)
        CWS = CWS.merge(hr2w_data, on='PWS.ID', how='left')
        print(f"After HR2W merge: {len(CWS)} records (should be same as before: {before_hr2w_count})")
        print(f"Systems with HR2W status: {len(CWS[CWS['SAFER.STATUS'].notna()])}")

        # Handle any remaining NaN values from merge (systems not found in HR2W data)
        na_count_after_merge = CWS['SAFER.STATUS'].isna().sum()
        if na_count_after_merge > 0:
            print(f"Found {na_count_after_merge} systems without SAFER.STATUS after merge")
            # Find which PWS.ID didn't match
            missing_ids = CWS[CWS['SAFER.STATUS'].isna()]['PWS.ID'].tolist()
            print(f"PWS.IDs without SAFER.STATUS: {missing_ids}")
            # Replace NaN with 'Not Assessed'
            CWS['SAFER.STATUS'] = CWS['SAFER.STATUS'].fillna('Not Assessed')
            print(f"Replaced {na_count_after_merge} missing SAFER.STATUS values with 'Not Assessed'")

    else:
        # If HR2W data couldn't be loaded, add empty column
        CWS['SAFER.STATUS'] = pd.NA
        print("HR2W column added with null values")
    
    # Write CSV file to Output Data directory
    print("Saving processed data...")
    CWS.to_csv(f"{output_folder}CWS_CA.csv", index=False)
    
    return CWS

if __name__ == "__main__":
    # Execute data processing
    try:
        processed_data = process_california_water_systems()
        print("Data processing completed successfully!")
        print(f"Processed dataset contains {len(processed_data)} water system records")
        
    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {str(e)}")
    except Exception as e:
        print(f"Error processing data: {str(e)}")