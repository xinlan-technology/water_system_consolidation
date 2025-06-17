# Explore the Consolidation Potential for Water Systems in California

This repository provides a comprehensive suite of Python scripts for analyzing and visualizing water system consolidation opportunities across California. The project integrates geospatial data, demographic attributes, and regulatory violations to evaluate physical and managerial consolidation potential through clustering algorithms.

## 📁 Project Structure

```
.
├── pws_centroid_extractor.py              # Extracts centroids from service area shapefiles
├── california_water_system_processor.py   # Loads and processes raw water system data
├── water_system_violation_analyzer.py     # Generates violation statistics by system size
├── california_water_violations_mapper.py  # Maps violations and HR2W risk systems
├── ca_water_population_analysis.py        # Creates population variation analysis and maps
├── california_county_demographics.py      # Plots county-level socioeconomic data
├── ca_road_network_downloader.py          # Downloads California road network from OpenStreetMap
├── pwsid_shortest_path_calculator.py      # Computes shortest path distances between water systems
├── precompute_clustering_table.py         # Precomputes optimal clustering numbers for thresholding
├── consolidation_sensitivity_analysis.py  # Runs sensitivity analysis across distance thresholds
├── consolidation_distance_analysis.py     # Visualizes consolidation outcomes and thresholds
├── case_study_consolidation_scenarios.py  # Provides the case studies using 10 km and 1 mile thresholds
├── california_consolidation_maps.py       # Generates maps and population change histograms for case study
├── consolidation_case_study_tables.py     # Generates summary tables of consolidation categories
├── Output Data/                           # Processed datasets and clustering tables (generated)
└── Output Figure/                         # Maps, plots, and visualizations (generated)
```

## 📋 Data Requirements

Before running the analysis, ensure the following input data files are placed in the `Input Data/` directory:

### Required Data Files
- **`cws_shape_file/`** - Folder containing water system service area shapefiles
  - `service_areas.shp` (and associated files: .shx, .dbf, .prj)
- **`detail_21_4_ca.csv`** - Water system details including primary source and service connections
- **`hauer_county_totpop_SSPs.csv`** - County-level population projection data
- **`HR2W_2022_12.csv`** - Human Right to Water (HR2W) assessment data
- **`summary_16_4_ca.csv`** - Water system summary data for 2016
- **`summary_21_4_ca.csv`** - Water system summary data for 2021
- **`violation_21_4_ca.csv`** - Water quality violation records

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- Required packages (install via pip):

```bash
pip install pandas numpy matplotlib geopandas scikit-learn scipy osmnx shapely
```

## 🚀 Usage

### Execution Sequence
Run the scripts in the following order for complete analysis:

```bash
# 1. Extract water system centroids from shapefiles
python pws_centroid_extractor.py

# 2. Process and integrate water system data
python california_water_system_processor.py

# 3. Analyze water system violations
python water_system_violation_analyzer.py

# 4. Create violation and HR2W risk maps
python california_water_violations_mapper.py

# 5. Analyze population variation patterns
python ca_water_population_analysis.py

# 6. Process county demographic data
python california_county_demographics.py

# 7. Download road network data
python ca_road_network_downloader.py
# Note: Road network data was downloaded on June 16, 2025
# Different download dates may result in slight variations in consolidation estimates

# 8. Calculate distances between systems
python pwsid_shortest_path_calculator.py

# 9. Precompute clustering numbers
python precompute_clustering_table.py

# 10. Run sensitivity analysis
python consolidation_sensitivity_analysis.py

# 11. Analyze distance thresholds vs consolidation potential
python consolidation_distance_analysis.py

# 12. Provide the case studies using 10 km and 1 mile thresholds
python case_study_consolidation_scenarios.py

# 13. Generate visualization maps
python california_consolidation_maps.py

# 14. Create summary tables
python consolidation_case_study_tables.py
```

### Output Structure
After execution, results will be organized as:
- **Output Data/** – Processed datasets, clustering tables, and analytical results
- **Output Figure/** – Generated maps, plots, and visualizations

## 📊 Key Features

- **Geospatial Analysis**: Road network-based distance calculations between water systems
- **Clustering Algorithms**: Automated identification of consolidation opportunities
- **Regulatory Compliance**: Analysis of violations and HR2W (Human Right to Water) risk systems
- **Sensitivity Analysis**: Evaluation of consolidation outcomes across multiple distance thresholds
- **Visualization Suite**: Comprehensive maps and statistical plots for decision support

## 📈 Applications

This toolkit supports water resource planners, policymakers, and researchers in:
- Identifying cost-effective consolidation opportunities
- Evaluating regulatory compliance scenarios
- Optimizing water system management strategies across California
