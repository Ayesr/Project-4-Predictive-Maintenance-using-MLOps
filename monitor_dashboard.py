# Task 2.2: Create Monitoring Dashboard
import requests
import time
import os
from datetime import datetime

API_URL = 'http://localhost:8000'

def clear_screen():
    # Clears terminal for a better dashboard experience
    os.system('cls' if os.name == 'nt' else 'clear')

def display_metrics():
    try:
        response = requests.get(f'{API_URL}/metrics', timeout=2)
        response.raise_for_status()
        metrics = response.json()
        
        clear_screen()
        print('=' * 40)
        print(f'METRICS DASHBOARD | {datetime.now().strftime("%H:%M:%S")}')
        print('=' * 40)
        print(f'Total Requests:      {metrics["total_requests"]}')
        print(f'Failures Predicted:  {metrics["failures_predicted"]}')
        print(f'Failure Rate:        {metrics["failure_rate"]:.1%}')
        print(f'Avg Latency:         {metrics["avg_latency_ms"]:.2f} ms')
        print(f'Errors:              {metrics["errors"]}')
        print('=' * 40)
        print('(Press Ctrl+C to stop)')
        
    except requests.exceptions.RequestException as e:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Connection Error: {e}')

if __name__ == '__main__':
    try:
        while True:
            display_metrics()
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nMonitoring stopped.')