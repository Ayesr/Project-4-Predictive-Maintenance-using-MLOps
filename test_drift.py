import numpy as np
import pandas as pd
from drift_detector import DriftDetector

def run_drift_test(detector: DriftDetector, current_data: pd.DataFrame, scenario_name: str):
    """Helper to run a drift check and print clean results."""
    print(f"--- Scenario: {scenario_name} ---")
    drift, results = detector.check_all_features(current_data)
    
    print(f"Drift Detected: {'YES' if drift else 'NO'}")
    if drift:
        for feature, info in results.items():
            if info['drift_detected']:
                print(f"  [!] {feature:12}: DRIFT | p-val: {info['p_value']:.4f}")
    print()

def main():
    # Setup
    np.random.seed(42)
    n_samples = 1000
    reference_data = pd.DataFrame({
        'temperature': np.random.normal(75, 15, n_samples),
        'vibration': np.random.normal(0.5, 0.2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples)
    })

    detector = DriftDetector(reference_data)

    # Scenario 1: No drift
    no_drift_data = pd.DataFrame({
        'temperature': np.random.normal(75, 15, 500),
        'vibration': np.random.normal(0.5, 0.2, 500),
        'pressure': np.random.normal(100, 20, 500)
    })
    run_drift_test(detector, no_drift_data, "No Drift (Same Distribution)")

    # Scenario 2: With drift
    drift_data = pd.DataFrame({
        'temperature': np.random.normal(95, 15, 500), # Shifted mean
        'vibration': np.random.normal(0.8, 0.2, 500), # Shifted mean
        'pressure': np.random.normal(100, 20, 500)
    })
    run_drift_test(detector, drift_data, "Drift Detected (Shifted Mean)")

    # Finalize
    detector.save_report()
    print("Report saved to drift_report.json")

if __name__ == '__main__':
    main()