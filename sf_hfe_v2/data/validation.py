"""
Data Validation and Cleaning Module
Ensures data quality before training
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging


class DataValidator:
"""
Validates and cleans input data

Checks:
- Missing values
- Data types
- Outliers (IQR method)
- Duplicates
- Value ranges
"""

def __init__(self):
self.logger = logging.getLogger("DataValidator")
self.validation_report = {}

def validate_and_clean(
self, 
data: np.ndarray, 
feature_names: Optional[List[str]] = None,
remove_outliers: bool = True,
fill_missing: bool = True,
remove_duplicates: bool = True
) -> Tuple[np.ndarray, Dict]:
"""
Complete validation and cleaning pipeline

Args:
data: Input data (numpy array or pandas DataFrame)
feature_names: Optional feature names
remove_outliers: Whether to remove outliers
fill_missing: Whether to fill missing values
remove_duplicates: Whether to remove duplicate rows

Returns:
cleaned_data: Cleaned numpy array
report: Validation report dictionary
"""
self.logger.info("Starting data validation and cleaning...")

# Convert to DataFrame for easier manipulation
if isinstance(data, np.ndarray):
if feature_names is None:
feature_names = [f"feature_{i}" for i in range(data.shape[1])]
df = pd.DataFrame(data, columns=feature_names)
elif isinstance(data, pd.DataFrame):
df = data.copy()
feature_names = df.columns.tolist()
else:
raise TypeError("Data must be numpy array or pandas DataFrame")

original_shape = df.shape
report = {
"original_shape": original_shape,
"original_rows": original_shape[0],
"original_cols": original_shape[1],
"issues_found": [],
"actions_taken": []
}

# 1. Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
report["missing_values"] = missing_counts[missing_counts > 0].to_dict()
report["issues_found"].append("Missing values detected")

if fill_missing:
# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
if df[col].isnull().any():
median_val = df[col].median()
df[col].fillna(median_val, inplace=True)

report["actions_taken"].append(f"Filled {missing_counts.sum()} missing values with median")
self.logger.info(f"Filled {missing_counts.sum()} missing values")

# 2. Check data types
dtypes_report = df.dtypes.astype(str).to_dict()
report["data_types"] = dtypes_report

# 3. Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
report["duplicates"] = int(duplicates)
report["issues_found"].append(f"{duplicates} duplicate rows found")

if remove_duplicates:
df = df.drop_duplicates()
report["actions_taken"].append(f"Removed {duplicates} duplicate rows")
self.logger.info(f"Removed {duplicates} duplicate rows")

# 4. Detect and remove outliers (IQR method)
if remove_outliers:
outlier_count = 0
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR # 3*IQR for less aggressive removal
upper_bound = Q3 + 3 * IQR

outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
if outliers > 0:
outlier_count += outliers
df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

if outlier_count > 0:
report["outliers_removed"] = int(outlier_count)
report["actions_taken"].append(f"Removed {outlier_count} outliers (IQR method)")
self.logger.info(f"Removed {outlier_count} outliers")

# 5. Basic statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
stats = df[numeric_cols].describe().to_dict()
report["statistics"] = stats

# 6. Check for infinite values
if df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
report["issues_found"].append("Infinite values detected")
# Replace inf with NaN, then fill with median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
if df[col].isnull().any():
df[col].fillna(df[col].median(), inplace=True)
report["actions_taken"].append("Replaced infinite values with median")

# Final shape
final_shape = df.shape
report["final_shape"] = final_shape
report["final_rows"] = final_shape[0]
report["final_cols"] = final_shape[1]
report["rows_removed"] = original_shape[0] - final_shape[0]
report["data_quality_score"] = self._compute_quality_score(df, report)

# Convert back to numpy array
cleaned_data = df.values

self.logger.info(f"Validation complete. Quality score: {report['data_quality_score']:.2f}/100")

return cleaned_data, report

def _compute_quality_score(self, df: pd.DataFrame, report: Dict) -> float:
"""
Compute overall data quality score (0-100)

Factors:
- No missing values: +30
- No duplicates: +20
- No outliers removed: +20
- Reasonable sample size: +30
"""
score = 0.0

# No missing values
if "missing_values" not in report:
score += 30
else:
missing_pct = sum(report["missing_values"].values()) / (df.shape[0] * df.shape[1])
score += max(0, 30 * (1 - missing_pct))

# No duplicates
if "duplicates" not in report:
score += 20
else:
dup_pct = report["duplicates"] / report["original_rows"]
score += max(0, 20 * (1 - dup_pct))

# Outliers
if "outliers_removed" not in report:
score += 20
else:
outlier_pct = report["outliers_removed"] / report["original_rows"]
score += max(0, 20 * (1 - min(outlier_pct, 0.2) / 0.2)) # Penalize if >20% outliers

# Sample size
if report["final_rows"] >= 100:
score += 30
elif report["final_rows"] >= 50:
score += 20
elif report["final_rows"] >= 20:
score += 10

return min(score, 100.0)

def validate_tensor(self, tensor: torch.Tensor) -> Tuple[bool, List[str]]:
"""
Quick validation for PyTorch tensors

Returns:
is_valid: Boolean
issues: List of issue descriptions
"""
issues = []

# Check for NaN
if torch.isnan(tensor).any():
issues.append("Tensor contains NaN values")

# Check for Inf
if torch.isinf(tensor).any():
issues.append("Tensor contains Inf values")

# Check shape
if len(tensor.shape) == 0:
issues.append("Tensor is 0-dimensional")

# Check if empty
if tensor.numel() == 0:
issues.append("Tensor is empty")

is_valid = len(issues) == 0

return is_valid, issues


class DataEntryProcessor:
"""
Process and validate user-entered data
"""

def __init__(self):
self.validator = DataValidator()
self.logger = logging.getLogger("DataEntryProcessor")

def process_csv(
self, 
file_path: str,
target_column: Optional[str] = None,
auto_clean: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
"""
Process CSV file

Args:
file_path: Path to CSV file
target_column: Name of target column (if any)
auto_clean: Whether to automatically clean data

Returns:
X: Feature data
y: Target data (or None)
report: Validation report
"""
self.logger.info(f"Processing CSV: {file_path}")

# Read CSV
df = pd.read_csv(file_path)

# Separate features and target
if target_column and target_column in df.columns:
y = df[target_column].values.reshape(-1, 1)
X = df.drop(columns=[target_column]).values
else:
X = df.values
y = None

# Validate and clean
if auto_clean:
X_clean, report = self.validator.validate_and_clean(X)
if y is not None:
y_clean, _ = self.validator.validate_and_clean(y, fill_missing=True)
return X_clean, y_clean, report
return X_clean, None, report
else:
report = {"status": "No cleaning performed"}
return X, y, report

def process_manual_input(
self,
data_dict: Dict[str, List],
auto_clean: bool = True
) -> Tuple[np.ndarray, Dict]:
"""
Process manually entered data

Args:
data_dict: Dictionary with feature names as keys, lists as values
auto_clean: Whether to automatically clean data

Returns:
data: Cleaned numpy array
report: Validation report
"""
self.logger.info("Processing manual input...")

# Convert to DataFrame
df = pd.DataFrame(data_dict)

# Validate and clean
if auto_clean:
cleaned_data, report = self.validator.validate_and_clean(df.values, list(df.columns))
return cleaned_data, report
else:
report = {"status": "No cleaning performed"}
return df.values, report

def process_numpy_array(
self,
data: np.ndarray,
auto_clean: bool = True
) -> Tuple[np.ndarray, Dict]:
"""
Process numpy array

Args:
data: Input numpy array
auto_clean: Whether to automatically clean

Returns:
cleaned_data: Cleaned array
report: Validation report
"""
if auto_clean:
return self.validator.validate_and_clean(data)
else:
report = {"status": "No cleaning performed"}
return data, report


def quick_validate(data: np.ndarray) -> Dict:
"""
Quick validation function for convenience

Args:
data: Input data

Returns:
report: Validation report
"""
validator = DataValidator()
_, report = validator.validate_and_clean(data, fill_missing=False, remove_outliers=False, remove_duplicates=False)
return report

