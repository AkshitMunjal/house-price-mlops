
# This is the target column that we cannot drop, regardless of the percentage of null values it contains.
IMPORTANT_COLUMNS = ['price']

# Threshold for dropping columns based on the percentage of null values. 
THRESHOLD = 0.3

# Columns that must exits and must not have nulls
CRITICAL_COLUMNS = ['location','size','total_sqft']

# Numerical columns
NUMERICAL_COLUMNS = ['bath']

# Categorical columns
CATEGORICAL_COLUMNS = ['balcony']