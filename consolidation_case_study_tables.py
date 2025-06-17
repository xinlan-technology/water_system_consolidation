# Water System Consolidation Analysis Tables Generator
# Generates consolidation analysis tables based on case study results

import pandas as pd
import numpy as np

def load_consolidation_data():
    """Load the consolidation case study results"""
    try:
        df = pd.read_csv('Output Data/CWS_CA_Case_Study_Results.csv')
        print(f"Loaded {len(df)} water systems with consolidation analysis")
        return df
    except FileNotFoundError:
        print("Error: CWS_CA_Case_Study_Results.csv not found")
        return None

def identify_system_categories(df):
    """Identify different categories of water systems based on violations and conditions"""
    categories = {}
    
    # Health-based violation systems
    categories['health-based violation'] = df[df['Health.violation'] > 0]
    
    # M&R (Monitoring and Reporting) violation systems
    categories['M&R violation'] = df[df['Monitoring.and.reporting.violation'] > 0]
    
    # Decreasing population systems (negative population change)
    categories['decreasing pop.'] = df[df['Population.Change'] < 0]
    
    # Systems with both violations and decreasing population
    categories['violation and decreasing pop.'] = df[
        ((df['Health.violation'] > 0) |
         (df['Monitoring.and.reporting.violation'] > 0)) &
        (df['Population.Change'] < 0)
    ]
    
    # HR2W (Human Right to Water) priority systems - only Failing status
    categories['HR2W'] = df[df['SAFER.STATUS'] == 'Failing']
    
    # All CWS
    categories['all CWS'] = df
    
    return categories

def identify_small_system_categories(df):
    """Identify categories for small water systems (up to 10,000 people served)"""
    # Filter for small systems (assuming population <= 10,000)
    small_df = df[df['Population.2021'] <= 10000]
    
    categories = {}
    
    # Health-based violation systems
    categories['health-based violation'] = small_df[small_df['Health.violation'] > 0]
    
    # M&R violation systems
    categories['M&R violation'] = small_df[small_df['Monitoring.and.reporting.violation'] > 0]
    
    # Decreasing population systems
    categories['decreasing pop.'] = small_df[small_df['Population.Change'] < 0]
    
    # Systems with both violations and decreasing population
    categories['violation and decreasing pop.'] = small_df[
        ((small_df['Health.violation'] > 0) |
         (small_df['Monitoring.and.reporting.violation'] > 0)) &
        (small_df['Population.Change'] < 0)
    ]
    
    # HR2W priority systems - only Failing status
    categories['HR2W'] = small_df[small_df['SAFER.STATUS'] == 'Failing']
    
    # All small CWS
    categories['all small CWS'] = small_df
    
    return categories

def count_consolidation_types(category_df):
    """Count systems by consolidation type for both physical and managerial"""
    physical_counts = {
        'DA': len(category_df[category_df['Physical_Consolidation_Type'] == 'Direct_Acquisition']),
        'JM': len(category_df[category_df['Physical_Consolidation_Type'] == 'Joint_Merger']),
        'BM': len(category_df[category_df['Physical_Consolidation_Type'] == 'Balanced_Merger'])
    }
    
    managerial_counts = {
        'DA': len(category_df[category_df['Managerial_Consolidation_Type'] == 'Direct_Acquisition']),
        'JM': len(category_df[category_df['Managerial_Consolidation_Type'] == 'Joint_Merger']),
        'BM': len(category_df[category_df['Managerial_Consolidation_Type'] == 'Balanced_Merger'])
    }
    
    return physical_counts, managerial_counts

def generate_all_cws_consolidation_analysis(df):
    """Generate All CWS Consolidation Analysis: All Community Water Systems eligible for consolidation"""
    print("\n" + "="*85)
    print("ALL COMMUNITY WATER SYSTEMS CONSOLIDATION ANALYSIS")
    print("Systems eligible for consolidation to address specified water issues")
    print("="*85)
    
    categories = identify_system_categories(df)
    
    # Print table header
    print(f"{'':30} | {'nb. of':>8} | {'physical consol.':>20} | {'managerial consol.':>20}")
    print(f"{'':30} | {'CWS':>8} | {'DA':>6} {'JM':>6} {'BM':>6} | {'DA':>6} {'JM':>6} {'BM':>6}")
    print("-" * 85)
    
    # Print each category
    for category_name, category_df in categories.items():
        physical_counts, managerial_counts = count_consolidation_types(category_df)
        
        print(f"{category_name:30} | {len(category_df):8} | "
              f"{physical_counts['DA']:6} {physical_counts['JM']:6} {physical_counts['BM']:6} | "
              f"{managerial_counts['DA']:6} {managerial_counts['JM']:6} {managerial_counts['BM']:6}")
    
    print("\nDA = Direct Acquisition; JM = Joint Merger; BM = Balanced Merger")
    print("HR2W = Human Right to Water priority systems")
    print("Physical consolidation: 1 mile threshold; Managerial consolidation: 10 km threshold")

def generate_small_cws_consolidation_analysis(df):
    """Generate Small CWS Consolidation Analysis: Small Community Water Systems eligible for consolidation"""
    print("\n" + "="*90)
    print("SMALL COMMUNITY WATER SYSTEMS CONSOLIDATION ANALYSIS")
    print("Systems (≤10,000 people) eligible for consolidation to address water issues")
    print("="*90)
    
    categories = identify_small_system_categories(df)
    
    # Print table header
    print(f"{'':30} | {'nb. of':>10} | {'physical consol.':>20} | {'managerial consol.':>20}")
    print(f"{'':30} | {'small CWS':>10} | {'DA':>6} {'JM':>6} {'BM':>6} | {'DA':>6} {'JM':>6} {'BM':>6}")
    print("-" * 90)
    
    # Print each category
    for category_name, category_df in categories.items():
        physical_counts, managerial_counts = count_consolidation_types(category_df)
        
        print(f"{category_name:30} | {len(category_df):10} | "
              f"{physical_counts['DA']:6} {physical_counts['JM']:6} {physical_counts['BM']:6} | "
              f"{managerial_counts['DA']:6} {managerial_counts['JM']:6} {managerial_counts['BM']:6}")
    
    print("\nDA = Direct Acquisition; JM = Joint Merger; BM = Balanced Merger")
    print("HR2W = Human Right to Water priority systems")
    print("Physical consolidation: 1 mile threshold; Managerial consolidation: 10 km threshold")

def main():
    """Main function to generate both consolidation analysis tables"""
    # Load the consolidation case study results
    df = load_consolidation_data()
    if df is None:
        return
    
    # Generate All CWS Consolidation Analysis
    generate_all_cws_consolidation_analysis(df)
    
    # Generate Small CWS Consolidation Analysis  
    generate_small_cws_consolidation_analysis(df)

    # Add consolidation summary statistics
    total_systems = len(df)
    physical_eligible = len(df[df['Physical_Consolidation_Type'].isin(['Direct_Acquisition', 'Joint_Merger', 'Balanced_Merger'])])
    managerial_eligible = len(df[df['Managerial_Consolidation_Type'].isin(['Direct_Acquisition', 'Joint_Merger', 'Balanced_Merger'])])
    print(f"\nTotal systems eligible for physical consolidation: {physical_eligible:,} ({physical_eligible/total_systems:.1%})")
    print(f"Total systems eligible for managerial consolidation: {managerial_eligible:,} ({managerial_eligible/total_systems:.1%})")
    
    print("\n" + "="*85)
    print("CONSOLIDATION ANALYSIS COMPLETED")
    print("="*85)

if __name__ == "__main__":
    main()