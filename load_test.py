# Task 3.2: Load Testing

import requests
import time
import random
from statistics import mean

API_URL = 'http://localhost:8000'
NUM_REQUESTS = 100

def generate_random_request():
    return {
        'temperature': random.uniform(60, 100),
        'vibration': random.uniform(0.3, 1.0),
        'pressure': random.uniform(80, 140),
        'rpm': random.uniform(1400, 1600),
        'age_days': random.randint(0, 365)
    }

def run_load_test():
    print(f'Starting load test: {NUM_REQUESTS} requests...')
    
    # Using a session is significantly faster for multiple requests to the same host
    with requests.Session() as session:
        start_time = time.time()
        successes = 0
        failures = 0
        latencies = []

        for i in range(1, NUM_REQUESTS + 1):
            try:
                req_start = time.time()
                response = session.post(f'{API_URL}/predict', json=generate_random_request(), timeout=5)
                req_time = (time.time() - req_start) * 1000
                
                if response.status_code == 200:
                    successes += 1
                    latencies.append(req_time)
                else:
                    failures += 1
            except requests.exceptions.RequestException:
                failures += 1
            
            if i % 20 == 0:
                print(f'Progress: {i}/{NUM_REQUESTS}')

    total_time = time.time() - start_time
    
    # Final Reporting
    print('\n' + '='*50)
    print('LOAD TEST RESULTS')
    print('='*50)
    print(f'Total Requests:    {NUM_REQUESTS}')
    print(f'Successful:        {successes}')
    print(f'Failed:            {failures}')
    print(f'Success Rate:      {successes/NUM_REQUESTS:.1%}')
    print(f'Total Time:        {total_time:.2f}s')
    print(f'Requests/sec:      {NUM_REQUESTS/total_time:.2f}')
    
    if latencies:
        print(f'Avg Latency:       {mean(latencies):.2f}ms')
        print(f'Min Latency:       {min(latencies):.2f}ms')
        print(f'Max Latency:       {max(latencies):.2f}ms')
    else:
        print('Avg Latency:       N/A (No successful requests)')
    print('='*50)

if __name__ == '__main__':
    run_load_test()