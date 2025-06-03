# California Water System Violation Analysis Script
# Generates comprehensive statistics on drinking water violations by system size
# Needs to read one file: CWS_CA.csv

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
        # Only process if there's actual HR2W data (not all null)
        hr2w_data_exists = df['SAFER.STATUS'].notna().any()
        
        if hr2w_data_exists:
            hr2w_counts = {}
            hr2w_statuses = ['Failing', 'At-Risk', 'Potentially At-Risk', 'Not At-Risk', 'Not Assessed']
            
            for status in hr2w_statuses:
                status_by_size = df[df['SAFER.STATUS'] == status].groupby('Size').size().to_dict()
                hr2w_counts[status.lower().replace(' ', ' ')] = status_by_size
                
                # Overall count
                hr2w_counts[f'{status.lower().replace(" ", " ")}_overall'] = len(df[df['SAFER.STATUS'] == status])
            
            results['hr2w_counts'] = hr2w_counts
            print(f"HR2W data available: {len(df[df['SAFER.STATUS'].notna()])} systems with status")
        else:
            print("HR2W column exists but no data available (all null values)")
    
    return results

def generate_summary_table(df):
    """
    Generate a formatted summary table
    
    Args:
        df (pandas.DataFrame): Water system data
        
    Returns:
        pandas.DataFrame: Formatted summary table
    """
    
    stats = calculate_violation_statistics(df)
    
    # Define the size order
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

def analyze_water_system_violations(input_file='Output Data/CWS_CA.csv'):
    """
    Analyze water system violations and display summary table
    Now simplified to only read one input file that contains all integrated data
    
    Args:
        input_file (str): Path to the processed water system data (with HR2W already integrated)
        
    Returns:
        pandas.DataFrame: Summary table for display
    """
    
    print("Loading integrated water system data...")
    
    try:
        # Read the processed CWS data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} water system records")
        
        # Check what data is available
        print("\nDataset summary:")
        print(f"Columns available: {list(df.columns)}")
        
        # Check HR2W data availability
        if 'SAFER.STATUS' in df.columns:
            hr2w_available = df['SAFER.STATUS'].notna().sum()
            print(f"HR2W data: {hr2w_available} systems with SAFER status out of {len(df)} total")
            if hr2w_available > 0:
                print("SAFER STATUS distribution:")
                print(df['SAFER.STATUS'].value_counts(dropna=False))
        else:
            print("No HR2W data found in dataset")
        
        # Generate comprehensive analysis
        print("\nCalculating violation statistics...")
        detailed_stats = calculate_violation_statistics(df)
        
        # Create formatted summary table
        print("Generating summary table...")
        summary_table = generate_summary_table(df)
        
        # Display summary
        print("\n" + "="*70)
        print("CALIFORNIA WATER SYSTEMS VIOLATION ANALYSIS SUMMARY")
        print("="*70)
        print(summary_table.to_string(index=False))
        print("="*70)
        
        return summary_table
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        print("Please make sure to run the data processing script first to generate CWS_CA.csv")
        return None
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the simplified analysis
    print("Starting California Water System Violation Analysis...")
    print("This script now only reads one file: CWS_CA.csv (with integrated HR2W data)")
    print("-" * 70)
    
    summary_table = analyze_water_system_violations()
    
    if summary_table is not None:
        print("\n✓ Analysis completed successfully!")
        print(f"Summary table has {len(summary_table)} rows and {len(summary_table.columns)} columns")
    else:
        print("\n✗ Analysis failed - please check error messages above")