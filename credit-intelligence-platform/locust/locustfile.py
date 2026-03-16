"""
locustfile.py

Load test for the /score endpoint. Measures p99 latency at 50 concurrent users.
Run with:
    locust -f locust/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 10 --run-time 60s --headless

The target is p99 < 120ms. Locust's built-in stats output after the run
will show this directly.
"""

import random
from locust import HttpUser, task, between


LOAN_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement","major_purchase", "medical"]


def random_application() -> dict:
    return {
        "applicant_id": f"load_test_{random.randint(1, 999999):06d}",
        "loan_amnt": random.uniform(3000, 35000),
        "int_rate": random.uniform(5.5, 28.9),
        "installment": random.uniform(80, 1200),
        "annual_inc": random.uniform(30000, 150000),
        "dti": random.uniform(0, 40),
        "fico_score": random.uniform(600, 800),
        "delinq_2yrs": random.randint(0, 5),
        "revol_util": random.uniform(0, 95),
        "total_acc": random.randint(3, 40),
        "open_acc": random.randint(2, 20),
        "pub_rec": random.randint(0, 3),
        "mort_acc": random.randint(0, 5),
        "emp_length": random.uniform(0, 10),
        "grade_enc": random.randint(0, 6),
        "sub_grade_enc": random.randint(0, 34),
        "term_months": random.choice([36, 60]),
        "funded_amnt": random.uniform(3000, 35000),
        "revol_bal": random.uniform(0, 50000),
        "pub_rec_bankruptcies": random.randint(0, 2),
        "unemployment_rate": random.uniform(3.5, 6.5),
        "cpi": random.uniform(290, 320),
        "fed_funds_rate": random.uniform(0.05, 5.5),
        "unemployment_mom_change": random.uniform(-0.2, 0.3),
        "cpi_yoy_pct": random.uniform(1.5, 4.0),
    }


class CreditScorerUser(HttpUser):
    wait_time = between(0.1, 0.5)  # ~2-10 req/sec per user → ~100-500 total at 50 users

    @task(10)
    def score_application(self):
        payload = random_application()
        with self.client.post(
            "/score",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "risk_score" not in data:
                    resp.failure("Missing risk_score in response")
            else:
                resp.failure(f"Non-200 status: {resp.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health")
