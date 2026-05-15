from locust import HttpUser, task, between


class ChurnApiUser(HttpUser):
    host = "http://13.51.205.181:443"
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        self.client.post(
            "/predict",
            json={
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
            },
        )

    @task(1)
    def health(self):
        self.client.get("/health")