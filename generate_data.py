"""Generate a realistic SYNTHETIC customer dataset for a streaming-service churn demo.

Churn is drawn from a probability that depends on customer behaviour plus random
noise, so relationships are meaningful but NOT deterministic.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42


def generate(n_customers: int = 4000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(34, 11, n_customers).round(), 18, 80).astype(int)
    gender = rng.choice(["Male", "Female", "Other"], n_customers, p=[0.48, 0.48, 0.04])
    plan = rng.choice(["Basic", "Standard", "Premium"], n_customers, p=[0.40, 0.35, 0.25])

    price_range = {"Basic": (149, 249), "Standard": (299, 499), "Premium": (499, 799)}
    monthly = np.array([rng.uniform(*price_range[p]) for p in plan]).round(0)

    tenure = np.clip(rng.exponential(18, n_customers).round(), 1, 72).astype(int)

    # Engagement (watch hours and logins are correlated with each other)
    engagement = rng.gamma(shape=2.5, scale=1.0, size=n_customers)          # latent engagement
    watch = np.clip(engagement * 4 + rng.normal(0, 3, n_customers), 0, 60).round(1)
    login = np.clip((engagement * 2 + rng.normal(0, 1.5, n_customers)).round(), 0, 21).astype(int)

    auto_renew = rng.choice(["Yes", "No"], n_customers, p=[0.68, 0.32])

    pay_lambda = np.where(auto_renew == "No", 0.55, 0.3)
    payment_failures = np.clip(rng.poisson(pay_lambda), 0, 6)
    complaints = np.clip(rng.poisson(0.5 + 0.15 * payment_failures), 0, 7)

    # Latent churn score (log-odds). Signs follow business intuition.
    z = (
        -1.0
        + 0.75 * payment_failures
        + 0.55 * complaints
        - 0.09 * watch
        - 0.14 * login
        - 0.035 * tenure
        + 0.90 * (auto_renew == "No")
        + 0.004 * (monthly - 400) / 10
        + 0.25 * (plan == "Basic")
        + 0.9
        + 0.9 * (tenure <= 6)                                   # early-life churn
        + 0.8 * ((watch < 5) & (login <= 2))                    # disengaged users
        + 0.5 * ((payment_failures >= 2) & (auto_renew == 'No'))  # payment + no auto-renew
    )
    z = z + rng.normal(0, 0.7, n_customers)        # realistic noise
    prob = 1 / (1 + np.exp(-z))
    churn = np.where(rng.random(n_customers) < prob, "Yes", "No")

    df = pd.DataFrame({
        "CustomerID": [f"CUST{100000 + i}" for i in range(n_customers)],
        "Age": age,
        "Gender": gender,
        "SubscriptionPlan": plan,
        "TenureMonths": tenure,
        "MonthlyCharges": monthly,
        "WatchHoursPerWeek": watch,
        "LoginFrequencyPerWeek": login,
        "PaymentFailures": payment_failures,
        "CustomerComplaints": complaints,
        "AutoRenewal": auto_renew,
        "Churn": churn,
    })

    # Inject a few missing values so the imputers have real work to do
    for col in ["Age", "WatchHoursPerWeek", "MonthlyCharges", "Gender"]:
        idx = rng.choice(n_customers, size=int(0.008 * n_customers), replace=False)
        df.loc[idx, col] = np.nan
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("data/customer_churn.csv", index=False)
    print(f"Saved {len(data)} rows to data/customer_churn.csv")
    print(data["Churn"].value_counts(normalize=True).round(3))
