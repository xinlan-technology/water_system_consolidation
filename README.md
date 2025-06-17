# California Water System Consolidation Analysis

This repository provides a comprehensive suite of Python scripts for analyzing and visualizing water system consolidation opportunities across California. The project integrates geospatial data, clustering algorithms, demographic attributes, and regulatory violations to evaluate both physical and managerial consolidation scenarios.

## 📁 Project Structure

```
.
├── ca_road_network_downloader.py          # Downloads California road network from OpenStreetMap
├── pws_centroid_extractor.py              # Extracts centroids from service area shapefiles
├── pwsid_shortest_path_calculator.py      # Computes shortest path distances between water systems
├── precompute_clustering_table.py         # Precomputes optimal clustering parameters for thresholding
├── case_study_consolidation_scenarios.py  # Assigns consolidation type to each system
├── consolidation_sensitivity_analysis.py  # Runs sensitivity analysis across distance thresholds
├── consolidation_distance_analysis.py     # Visualizes consolidation outcomes and thresholds
├── california_consolidation_maps.py       # Generates maps and population histograms for case study
├── california_water_system_processor.py   # Loads and aligns raw water system data
├── california_water_violations_mapper.py  # Maps violations and HR2W risk systems
├── california_county_demographics.py      # Merges county-level socioeconomic data
├── consolidation_case_study_tables.py     # Generates summary tables of consolidation categories
├── Output Data/                           # Processed datasets and clustering tables (generated)
└── Output Figure/                         # Maps, plots, and visualizations (generated)
```

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
# 1. Download road network data
python ca_road_network_downloader.py

# 2. Extract water system centroids
python pws_centroid_extractor.py

# 3. Calculate distances between systems
python pwsid_shortest_path_calculator.py

# 4. Precompute clustering parameters
python precompute_clustering_table.py

# 5. Assign consolidation scenarios
python case_study_consolidation_scenarios.py

# 6. Run sensitivity analysis
python consolidation_sensitivity_analysis.py

# 7. Analyze distance thresholds
python consolidation_distance_analysis.py

# 8. Generate visualization maps
python california_consolidation_maps.py

# 9. Create summary tables
python consolidation_case_study_tables.py
```

### Output Structure
After execution, results will be organized as:
- **Output Data/** – Processed datasets, clustering tables, and analytical results
- **Output Figure/** – Generated maps, plots, and visualizations

## 📊 Key Features

- **Geospatial Analysis**: Road network-based distance calculations between water systems
- **Clustering Algorithms**: Automated identification of consolidation opportunities
- **Demographic Integration**: Incorporation of socioeconomic factors in consolidation decisions
- **Regulatory Compliance**: Analysis of violations and HR2W (Human Right to Water) risk systems
- **Sensitivity Analysis**: Evaluation of consolidation outcomes across multiple distance thresholds
- **Visualization Suite**: Comprehensive maps and statistical plots for decision support

## 📈 Applications

This toolkit supports water resource planners, policymakers, and researchers in:
- Identifying cost-effective consolidation opportunities
- Evaluating regulatory compliance scenarios
- Assessing demographic impacts of consolidation policies
- Optimizing water system management strategies across California
