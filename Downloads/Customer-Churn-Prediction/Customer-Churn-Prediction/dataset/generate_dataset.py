"""
Realistic Telecom Customer Churn Dataset Generator
Generates industrial-grade synthetic data mimicking real telecom patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 7043  # Match real Telco dataset size


def generate_churn_dataset(n_samples=N, output_path=None):
    """Generate a realistic telecom churn dataset."""

    # Customer demographics
    customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.483, 0.517])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.299, 0.701])

    # Service tenure (months)
    tenure = np.random.choice(range(0, 73), size=n_samples)
    # Bias: churned customers tend to have lower tenure
    tenure_churn_bias = np.where(tenure < 12, 0.45, np.where(tenure < 24, 0.28, 0.12))

    # Phone & Internet Services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.904, 0.096])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.421, 0.579]),
    )

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.343, 0.439, 0.218],
    )

    online_security = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.284, 0.716]),
    )
    online_backup = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.342, 0.658]),
    )
    device_protection = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.343, 0.657]),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.290, 0.710]),
    )
    streaming_tv = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.385, 0.615]),
    )
    streaming_movies = np.where(
        internet_service == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.388, 0.612]),
    )

    # Contract & Billing
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.550, 0.211, 0.239],
    )
    paperless_billing = np.random.choice(
        ["Yes", "No"], size=n_samples, p=[0.592, 0.408]
    )
    payment_method = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.336, 0.229, 0.218, 0.217],
    )

    # Monthly charges (fiber users pay more)
    monthly_charges = np.where(
        internet_service == "Fiber optic",
        np.random.normal(80, 15, n_samples).clip(50, 120),
        np.where(
            internet_service == "DSL",
            np.random.normal(55, 12, n_samples).clip(25, 90),
            np.random.normal(25, 8, n_samples).clip(15, 55),
        ),
    ).round(2)

    # Total charges correlated with tenure
    total_charges = (monthly_charges * tenure + np.random.normal(0, 50, n_samples)).clip(0).round(2)
    total_charges = np.where(tenure == 0, monthly_charges, total_charges)

    # Support tickets
    num_support_tickets = np.random.poisson(1.5, n_samples).clip(0, 10)

    # Churn label — engineered with realistic logic
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == "Month-to-month", 0.25, 0.0)
    churn_prob += np.where(contract == "One year", 0.08, 0.0)
    churn_prob += np.where(internet_service == "Fiber optic", 0.12, 0.0)
    churn_prob += np.where(payment_method == "Electronic check", 0.10, 0.0)
    churn_prob += np.where(online_security == "No", 0.08, 0.0)
    churn_prob += np.where(tech_support == "No", 0.06, 0.0)
    churn_prob += np.where(tenure < 6, 0.15, np.where(tenure < 12, 0.08, 0.0))
    churn_prob += np.where(monthly_charges > 75, 0.10, 0.0)
    churn_prob += (num_support_tickets * 0.03)
    churn_prob += np.where(senior_citizen == 1, 0.05, 0.0)
    churn_prob = churn_prob.clip(0.02, 0.92)

    churn_labels = np.random.binomial(1, churn_prob)
    churn = np.where(churn_labels == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "NumSupportTickets": num_support_tickets,
        "Churn": churn,
    })

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path} — {len(df)} rows, churn rate: {(df['Churn']=='Yes').mean():.1%}")

    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "telco_churn.csv"
    generate_churn_dataset(output_path=out)
