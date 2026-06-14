import pytest
import pandas as pd
import numpy as np

# Fixture to provide standard test data
@pytest.fixture
def sample_input():
    return pd.DataFrame({
        'temperature': [75.0],
        'vibration': [0.5],
        'pressure': [100.0],
        'rpm': [1500.0],
        'age_days': [100]
    })

def test_data_generation_constraints():
    """Verify generated data has expected size and no missing values."""
    n_samples = 100
    data = pd.DataFrame({
        'temperature': np.random.normal(75, 15, n_samples),
        'vibration': np.random.normal(0.5, 0.2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples)
    })
    
    assert len(data) == n_samples, f"Expected {n_samples} samples, got {len(data)}"
    assert data.isnull().sum().sum() == 0, "Data contains unexpected null values"
    
    # Ensure values are within reasonable physical bounds (optional but recommended)
    assert (data['temperature'] > 0).all(), "Negative temperatures are not allowed"

def test_model_input_shape(sample_input):
    """Verify input shape matches expected dimensions for a single prediction."""
    expected_shape = (1, 5)
    assert sample_input.shape == expected_shape, f"Expected shape {expected_shape}, got {sample_input.shape}"

def test_model_input_columns(sample_input):
    """Ensure the input DataFrame contains all required features."""
    required_features = {'temperature', 'vibration', 'pressure', 'rpm', 'age_days'}
    assert required_features.issubset(sample_input.columns), "Missing required input columns"

def test_invalid_input_type():
    """Example test to ensure code handles invalid input types (if you implement validation)."""
    with pytest.raises(ValueError):
        # Assuming your function raises ValueError for non-numeric data
        pd.DataFrame({'temperature': ['high'], 'vibration': [0.5]})