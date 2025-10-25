# Simulated Data Files

## **Simulated Test Data in `tests/simulated_data/`**

This folder contains all simulated data files used for testing the federated learning system and OMEO expert system.

### **Data Files:**

#### **1. `test_data.csv` (3,517 bytes)**
- **Purpose**: Basic test data for system functionality testing
- **Format**: CSV with numerical features
- **Usage**: Used by basic system tests and integration tests
- **Generated**: Synthetic data for testing core functionality

#### **2. `test_omoe_data.csv` (9,811 bytes)**
- **Purpose**: OMEO expert system test data
- **Format**: CSV with multiple variable types
- **Usage**: Used by OMEO expert system tests
- **Generated**: Synthetic data designed for 30-expert processing
- **Features**: Mixed data types suitable for cross-expert reasoning

#### **3. `test_omoe_data_sf_hfe.csv` (9,828 bytes)**
- **Purpose**: Additional OMEO test data from sf_hfe_v2 directory
- **Format**: CSV with comprehensive feature set
- **Usage**: Used by advanced OMEO tests and federated learning integration
- **Generated**: Synthetic data for testing complete system integration
- **Features**: Multi-domain compatible data structure

### **Data Characteristics:**

#### **Data Types:**
- **Numerical**: Continuous variables for statistical analysis
- **Categorical**: Discrete variables for relational analysis
- **Mixed**: Combination of numerical and categorical features

#### **Domain Compatibility:**
- **Economics**: GDP, inflation, unemployment, interest rates
- **Agriculture**: Rainfall, temperature, crop yield, soil conditions
- **Education**: Test scores, attendance, demographics, performance metrics

#### **Expert System Compatibility:**
- **Structural Experts**: Schema mapping, data types, missing values
- **Statistical Experts**: Correlations, distributions, anomalies
- **Temporal Experts**: Trends, seasonality, drift patterns
- **Relational Experts**: Graph structures, influence networks
- **Causal Experts**: Causal relationships, counterfactuals
- **Semantic Experts**: Context, domain ontologies
- **Cognitive Experts**: Reasoning, simulation, forecasting

### **Usage in Tests:**

#### **Basic System Tests:**
- `test_system.py` - Uses `test_data.csv`
- `test_integration.py` - Uses `test_data.csv`

#### **OMEO Expert Tests:**
- `test_full_omoe.py` - Uses `test_omoe_data.csv`
- `test_30_experts.py` - Uses `test_omoe_data.csv`

#### **Federated Learning Tests:**
- `test_complete_federated_system.py` - Generates its own simulated data
- `test_actual_federated_learning.py` - Uses simulated client data

#### **Simulation Tests:**
- `test_simulation_engine.py` - Uses `test_data.csv`
- `test_3d_simulation.py` - Uses `test_omoe_data.csv`

### **Data Generation:**

All simulated data files are generated with:
- **Realistic correlations** between variables
- **Proper data distributions** for statistical analysis
- **Domain-specific patterns** for expert system processing
- **Mixed data types** for comprehensive testing
- **Sufficient sample sizes** for reliable analysis

### **File Organization:**

```
tests/
├── simulated_data/ # All simulated data files
│ ├── test_data.csv # Basic test data
│ ├── test_omoe_data.csv # OMEO test data
│ └── test_omoe_data_sf_hfe.csv # Advanced OMEO data
├── test_*.py # Test scripts
└── ...
```

### **Notes:**

- **Real Data**: `Market_Prices.csv` remains in root directory (real market data)
- **Generated Data**: All files in this folder are synthetically generated
- **Test-Specific**: Each file is designed for specific test scenarios
- **Domain-Agnostic**: Data can be used across different domain tests
- **Expert-Compatible**: All data is compatible with the 30-expert system

This organization keeps all simulated test data centralized and easily accessible for testing purposes while maintaining clear separation from real data files.
