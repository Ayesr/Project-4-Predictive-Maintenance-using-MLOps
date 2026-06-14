# PART 3: TEST & VALIDATE
# Task 3.1: Test API Endpoints

import requests
import json

API_URL = 'http://localhost:8000'

def run_tests():
    # 1. Test Health Check
    print(f"{'='*10} Testing /health endpoint {'='*10}")
    try:
        response = requests.get(f'{API_URL}/health', timeout=3)
        response.raise_for_status()
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}\n")
        return

    # 2. Test Predictions
    test_cases = [
        {
            'name': 'Normal Operation',
            'data': {'temperature': 70, 'vibration': 0.4, 'pressure': 95, 'rpm': 1500, 'age_days': 100}
        },
        {
            'name': 'High Risk',
            'data': {'temperature': 95, 'vibration': 0.9, 'pressure': 135, 'rpm': 1500, 'age_days': 320}
        }
    ]

    print(f"{'='*10} Testing /predict endpoint {'='*10}")
    for test in test_cases:
        print(f"Scenario: {test['name']}")
        try:
            response = requests.post(f'{API_URL}/predict', json=test['data'], timeout=3)
            response.raise_for_status()
            result = response.json()
            
            # Formatting results
            print(f"  [✓] Success")
            print(f"      Will Fail:   {result.get('will_fail')}")
            print(f"      Probability: {result.get('probability', 0):.3f}")
            print(f"      Latency:     {result.get('latency_ms', 0):.2f} ms\n")
            
        except requests.exceptions.RequestException as e:
            print(f"  [!] Request failed for {test['name']}: {e}\n")

if __name__ == '__main__':
    run_tests()