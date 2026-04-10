import time
import requests
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

# Researcher-grade performance profiling
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CreditAPIBenchmark:
    """
    High-concurrency latency profiler for the Credit Intelligence API.
    Used to verify p99 targets under simulated production load.
    """
    
    def __init__(self, host="http://localhost:8000"):
        self.host = host
        self.sample_payload = {
            "loan_amnt": 15000,
            "int_rate": 14.5,
            "installment": 520.0,
            "annual_inc": 72000,
            "dti": 18.5,
            "fico_score": 680,
            "emp_length": 4.0
        }

    def single_request(self):
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.host}/score", json=self.sample_payload, timeout=2)
            resp.raise_for_status()
            return (time.perf_counter() - start) * 1000
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def run_benchmark(self, n_requests=500, concurrent_users=10):
        logger.info(f"Starting benchmark: {n_requests} requests with {concurrent_users} concurrency...")
        
        latencies = []
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            results = list(executor.map(lambda _: self.single_request(), range(n_requests)))
            
        latencies = [l for l in results if l is not None]
        
        if not latencies:
            logger.error("No successful requests to profile.")
            return

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg = np.mean(latencies)
        
        print("\n" + "="*40)
        print("LATENCY PROFILE REPORT")
        print("="*40)
        print(f"Total Requests: {len(latencies)}")
        print(f"Average:        {avg:.2f} ms")
        print(f"p50 (Median):   {p50:.2f} ms")
        print(f"p95:            {p95:.2f} ms")
        print(f"p99:            {p99:.2f} ms")
        print(f"Throughput:     {len(latencies) / (np.sum(latencies)/1000):.2f} req/sec (approx)")
        print("="*40)
        
        if p99 < 100:
            logger.info("SLA VERIFIED: p99 latency is within the <100ms target.")
        else:
            logger.warning("SLA BREACH: p99 latency exceeded 100ms.")

if __name__ == "__main__":
    # Note: Requires the API service to be running (docker compose up -d)
    profiler = CreditAPIBenchmark()
    profiler.run_benchmark(n_requests=200, concurrent_users=5)
