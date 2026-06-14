import numpy as np
import pandas as pd
from scipy import stats
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """Initialize with training data distribution."""
        if reference_data.empty:
            raise ValueError("Reference data cannot be empty.")
        
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_history: List[Dict[str, Any]] = []

    def detect_drift_ks(self, current_data: pd.DataFrame, feature: str) -> Tuple[bool, float, float]:
        """Performs KS test; handles potential data errors gracefully."""
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            raise ValueError(f"Feature '{feature}' not found in data.")
            
        ref_values = self.reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        # Avoid test if samples are too small
        if len(ref_values) < 2 or len(curr_values) < 2:
            return False, 1.0, 0.0

        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
        return bool(p_value < self.threshold), float(p_value), float(statistic)

    def check_all_features(self, current_data: pd.DataFrame, apply_bonferroni: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Check drift across all features. 
        Bonferroni correction reduces false positives when testing multiple features.
        """
        results = {}
        n_features = len(self.reference_data.columns)
        # Adjust threshold if applying Bonferroni correction
        adjusted_threshold = self.threshold / n_features if apply_bonferroni else self.threshold
        
        drift_detected_any = False

        for feature in self.reference_data.columns:
            drift, p_val, stat = self.detect_drift_ks(current_data, feature)
            
            # Re-evaluate drift with potentially adjusted threshold
            drift = p_val < adjusted_threshold
            
            results[feature] = {
                'drift_detected': drift,
                'p_value': p_val,
                'statistic': stat
            }
            if drift:
                drift_detected_any = True

        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected_any,
            'features': results
        })
        
        return drift_detected_any, results

    def save_report(self, filename: str = 'drift_report.json') -> None:
        """Save history; handles potential IO errors."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.drift_history, f, indent=2)
        except IOError as e:
            print(f"Error saving drift report: {e}")