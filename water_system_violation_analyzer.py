# California Water System Violation Analysis Script
# Generates comprehensive statistics on drinking water violations by system size

import pandas as pd
import numpy as np

def create_size_categories(population):
    """
    Categorize water systems by service population size
    
    Args:
        population (pandas.Series): Population served in 2021
        
    Returns:
        pandas.Series: Size categories
    """
    conditions = [
        (population > 0) & (population <= 500),
        (population > 500) & (population <= 3300),
        (population > 3300) & (population <= 10000),
        (population > 10000) & (population <= 100000),
        (population > 100000)
    ]
    
    # Create size categories
    choices = ['0-500', '501-3300', '3301-10000', '10001-100000', '>100000']
    
    return pd.Series(np.select(conditions, choices, default='Unknown'), index=population.index)

def calculate_violation_statistics(df):
    """
    Calculate comprehensive violation statistics by water system size
    
    Args:
        df (pandas.DataFrame): Water system data with violation information
        
    Returns:
        dict: Dictionary containing various statistical summaries
    """
    
    # Create size categories
    df['Size'] = create_size_categories(df['Population.2021'])
    
    # Define violation columns
    violation_cols = {
        'health-based': 'Health.violation',
        'monitoring or reporting': 'Monitoring.and.reporting.violation', 
        'maximum contaminant levels': 'Maximum.contaminant.levels.violation',
        'treatment technique': 'Treatment.technique.violation'
    }
    
    results = {}
    
    # 1. System counts by size
    system_counts = df.groupby('Size').size().to_dict()
    results['system_counts'] = system_counts
    results['total_systems'] = len(df)
    
    # 2. Average violations per system (5-year period 2017-2021)
    avg_violations = {}
    for viol_name, col_name in violation_cols.items():
        avg_by_size = df.groupby('Size')[col_name].mean().to_dict()
        avg_violations[viol_name] = avg_by_size
        avg_violations[f'{viol_name}_overall'] = df[col_name].mean()
    
    results['avg_violations'] = avg_violations
    
    # 3. Fraction of systems with at least one violation
    violation_fractions = {}
    for viol_name, col_name in violation_cols.items():
        # Systems with at least one violation by size
        systems_with_violations = df[df[col_name] > 0].groupby('Size').size().to_dict()
        fractions_by_size = {}
        
        for size in system_counts.keys():
            violations_count = systems_with_violations.get(size, 0)
            total_count = system_counts[size]
            fractions_by_size[size] = violations_count / total_count if total_count > 0 else 0
        
        violation_fractions[viol_name] = fractions_by_size
        
        # Overall fraction
        total_with_violations = len(df[df[col_name] > 0])
        violation_fractions[f'{viol_name}_overall'] = total_with_violations / len(df)
    
    results['violation_fractions'] = violation_fractions
    
    # 4. HR2W (Human Right to Water) status counts by size
    if 'SAFER.STATUS' in df.columns:
        hr2w_counts = {}
        hr2w_statuses = ['Failing', 'At-Risk', 'Potentially At-Risk', 'Not At-Risk', 'Not Assessed']
        
        for status in hr2w_statuses:
            status_by_size = df[df['SAFER.STATUS'] == status].groupby('Size').size().to_dict()
            hr2w_counts[status.lower().replace(' ', ' ')] = status_by_size
            
            # Overall count
            hr2w_counts[f'{status.lower().replace(" ", " ")}_overall'] = len(df[df['SAFER.STATUS'] == status])
        
        results['hr2w_counts'] = hr2w_counts
    
    return results

def load_hr2w_data(hr2w_file='Input Data/HR2W_2022_12.csv'):
    """
    Load and process HR2W (Human Right to Water) data
    
    Args:
        hr2w_file (str): Path to HR2W list file
        
    Returns:
        pandas.DataFrame: Processed HR2W data
    """
    try:
        # Read HR2W data
        hr2w = pd.read_csv(hr2w_file)
        
        # Select only the needed columns
        hr2w_list = hr2w[["PWSID", "SAFER STATUS"]].copy()
        
        # Rename to standardize (convert space to dot for consistency with rest of code)
        hr2w_list.columns = ['PWSID', 'SAFER.STATUS']
        
        # Ensure PWSID is string for consistent merging
        hr2w_list['PWSID'] = hr2w_list['PWSID'].astype(str)
        
        print(f"Loaded {len(hr2w_list)} HR2W records")
        print(f"SAFER STATUS distribution:")
        print(hr2w_list['SAFER.STATUS'].value_counts())
        
        return hr2w_list
        
    except FileNotFoundError:
        print(f"Warning: Could not find HR2W file {hr2w_file}")
        print("HR2W analysis will be skipped")
        return None
    
    except Exception as e:
        print(f"Error loading HR2W data: {str(e)}")
        return None

def generate_summary_table(df):
    """
    Generate a formatted summary table
    
    Args:
        df (pandas.DataFrame): Water system data
        
    Returns:
        pandas.DataFrame: Formatted summary table
    """
    
    stats = calculate_violation_statistics(df)
    
    # Define the size order to match R output
    size_order = ['0-500', '501-3300', '3301-10000', '10001-100000', '>100000']
    
    # Create the summary table
    table_data = []
    
    # System amounts row
    system_row = ['system amount in 2021', stats['total_systems']]
    for size in size_order:
        system_row.append(stats['system_counts'].get(size, 0))
    table_data.append(system_row)
    
    # Empty row for section separation
    table_data.append(['', '', '', '', '', '', ''])
    table_data.append(['5-year average number of violations per system (2017-2021)', '', '', '', '', '', ''])
    
    # Average violations rows
    violation_types = [
        ('health-based', 'health-based'),
        ('monitoring or reporting', 'monitoring or reporting'),
        ('maximum contaminant levels', 'maximum contaminant levels'),
        ('treatment technique', 'treatment technique')
    ]
    
    for display_name, key_name in violation_types:
        row = [display_name, round(stats['avg_violations'][f'{key_name}_overall'], 1)]
        for size in size_order:
            avg_val = stats['avg_violations'][key_name].get(size, 0)
            row.append(round(avg_val, 1))
        table_data.append(row)
    
    # Empty row for section separation
    table_data.append(['', '', '', '', '', '', ''])
    table_data.append(['fraction of systems with at least one violation (2017-2021)', '', '', '', '', '', ''])
    
    # Fraction rows
    for display_name, key_name in violation_types:
        row = [display_name, f"{stats['violation_fractions'][f'{key_name}_overall']:.0%}"]
        for size in size_order:
            fraction_val = stats['violation_fractions'][key_name].get(size, 0)
            row.append(f"{fraction_val:.0%}")
        table_data.append(row)
    
    # Add HR2W section if data is available
    if 'hr2w_counts' in stats:
        table_data.append(['', '', '', '', '', '', ''])
        table_data.append(['system amount on human right to water (HR2W) list in 2022', '', '', '', '', '', ''])
        
        hr2w_statuses = [
            ('failing', 'failing'),
            ('at-risk', 'at-risk'), 
            ('potentially at-risk', 'potentially at-risk'),
            ('not at-risk', 'not at-risk'),
            ('not assessed', 'not assessed')
        ]
        
        for display_name, key_name in hr2w_statuses:
            row = [display_name, stats['hr2w_counts'].get(f'{key_name}_overall', 0)]
            for size in size_order:
                hr2w_val = stats['hr2w_counts'][key_name].get(size, 0)
                row.append(hr2w_val)
            table_data.append(row)
    
    # Create DataFrame with updated column headers
    columns = ['', 'all CWS', '0-500', '501-3300', '3301-10000', '10001-100000', '>100000']
    summary_table = pd.DataFrame(table_data, columns=columns)
    
    return summary_table

def analyze_water_system_violations(input_file='Output Data/CWS_CA.csv', hr2w_file='Input Data/HR2W_2022_12.csv'):
    """
    Analyze water system violations and display summary table
    
    Args:
        input_file (str): Path to the processed water system data
        hr2w_file (str): Path to the HR2W list file
        
    Returns:
        pandas.DataFrame: Summary table for display only
    """
    
    print("Loading water system data...")
    
    try:
        # Read the processed CWS data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} water system records")
        
        # Load and merge HR2W data
        hr2w_data = load_hr2w_data(hr2w_file)
        if hr2w_data is not None:
            # Ensure PWS.ID is string for consistent merging
            df['PWS.ID'] = df['PWS.ID'].astype(str)
            
            # Merge with HR2W data (left join to keep all CWS records)
            df = df.merge(hr2w_data, left_on='PWS.ID', right_on='PWSID', how='left')
            print(f"Merged with HR2W data: {len(df[df['SAFER.STATUS'].notna()])} systems have HR2W status")
        
        # Generate comprehensive analysis
        print("Calculating violation statistics...")
        detailed_stats = calculate_violation_statistics(df)
        
        # Create formatted summary table
        print("Generating summary table...")
        summary_table = generate_summary_table(df)
        
        # Display summary
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(summary_table.to_string(index=False))
        print("="*60)
        
        return summary_table
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        return None
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the simplified analysis
    summary_table = analyze_water_system_violations()
    
    if summary_table is not None:
        print("\n✓ Analysis completed!")