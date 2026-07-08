"""
Production use-case templates — business context, demo datasets, and inference actions.
"""
from __future__ import annotations

import os
import shutil

USE_CASES: dict[str, dict] = {
    "customer_churn": {
        "id": "customer_churn",
        "name": "Customer Churn Prediction",
        "industry": "Telecom / SaaS / Subscription",
        "description": (
            "Predict which customers are likely to cancel their subscription so retention "
            "teams can intervene before revenue is lost."
        ),
        "business_goal": "Reduce monthly churn rate by prioritizing high-risk accounts for outreach.",
        "dataset_file": "data/customer_churn.csv",
        "target_col": "Churn",
        "positive_class": "Yes",
        "problem_type": "classification",
        "id_columns": ["customer_id"],
        "risk_thresholds": {"high": 0.65, "medium": 0.35},
        "actions": {
            "high": "Immediate retention call + discount or plan upgrade offer",
            "medium": "Proactive check-in email and usage review within 7 days",
            "low": "Monitor in next billing cycle; include in nurture campaign",
        },
        "kpis": ["Churn rate", "Retention save rate", "Cost per saved customer"],
        "probability_label": "churn_probability",
        "report_prefix": "retention_report",
        "high_risk_label": "Churn risk",
    },
    "lead_scoring": {
        "id": "lead_scoring",
        "name": "Sales Lead Scoring",
        "industry": "B2B SaaS / Enterprise Sales",
        "description": (
            "Score inbound leads by conversion likelihood so sales teams focus on "
            "high-intent prospects first."
        ),
        "business_goal": "Increase win rate by routing hot leads to senior reps immediately.",
        "dataset_file": "data/lead_scoring.csv",
        "target_col": "Converted",
        "positive_class": "Yes",
        "problem_type": "classification",
        "id_columns": ["lead_id"],
        "risk_thresholds": {"high": 0.70, "medium": 0.40},
        "actions": {
            "high": "Assign to senior AE — same-day personalized outreach",
            "medium": "SDR follow-up within 48 hours + targeted case study",
            "low": "Add to automated nurture sequence; no immediate sales call",
        },
        "kpis": ["Conversion rate", "Lead-to-opportunity rate", "Sales cycle length"],
        "probability_label": "conversion_probability",
        "report_prefix": "lead_scoring_report",
        "high_risk_label": "Conversion score",
    },
    "fraud_detection": {
        "id": "fraud_detection",
        "name": "Payment Fraud Detection",
        "industry": "Banking / Fintech / E-commerce",
        "description": (
            "Flag suspicious transactions before they settle so fraud teams can block "
            "or verify high-risk payments in real time."
        ),
        "business_goal": "Reduce fraud losses while minimizing false declines on legitimate purchases.",
        "dataset_file": "data/fraud_detection.csv",
        "target_col": "IsFraud",
        "positive_class": "Yes",
        "problem_type": "classification",
        "id_columns": ["transaction_id"],
        "risk_thresholds": {"high": 0.75, "medium": 0.45},
        "actions": {
            "high": "Block transaction and alert fraud team immediately",
            "medium": "Step-up verification (OTP / call back) before approval",
            "low": "Approve with standard monitoring",
        },
        "kpis": ["Fraud catch rate", "False positive rate", "Chargeback volume"],
        "probability_label": "fraud_probability",
        "report_prefix": "fraud_alert_report",
        "high_risk_label": "Fraud risk",
    },
    "loan_default": {
        "id": "loan_default",
        "name": "Loan Default Risk",
        "industry": "Banking / Lending / Credit",
        "description": (
            "Predict which loan applicants are likely to default so underwriters can "
            "adjust terms, require collateral, or decline risky applications."
        ),
        "business_goal": "Cut default rate while keeping approval rates healthy for qualified borrowers.",
        "dataset_file": "data/loan_default.csv",
        "target_col": "Default",
        "positive_class": "Yes",
        "problem_type": "classification",
        "id_columns": ["applicant_id"],
        "risk_thresholds": {"high": 0.70, "medium": 0.40},
        "actions": {
            "high": "Decline or require co-signer + higher interest rate",
            "medium": "Manual underwriter review within 24 hours",
            "low": "Auto-approve with standard terms",
        },
        "kpis": ["Default rate", "Approval rate", "Loss given default"],
        "probability_label": "default_probability",
        "report_prefix": "loan_risk_report",
        "high_risk_label": "Default risk",
    },
}

SAMPLE_BATCHES: dict[str, list[dict]] = {
    "customer_churn": [
        {
            "tenure_months": 4,
            "monthly_charges": 95.0,
            "total_charges": 280.0,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "payment_method": "Electronic check",
            "senior_citizen": 0,
        },
        {
            "tenure_months": 36,
            "monthly_charges": 45.0,
            "total_charges": 1620.0,
            "contract_type": "Two year",
            "internet_service": "DSL",
            "tech_support": "Yes",
            "payment_method": "Bank transfer",
            "senior_citizen": 0,
        },
    ],
    "lead_scoring": [
        {
            "company_size": "201-1000",
            "industry": "Tech",
            "page_views_30d": 28,
            "demo_attended": "Yes",
            "email_opens_30d": 12,
            "sales_calls": 3,
            "days_in_pipeline": 7,
        },
        {
            "company_size": "1-10",
            "industry": "Retail",
            "page_views_30d": 3,
            "demo_attended": "No",
            "email_opens_30d": 1,
            "sales_calls": 0,
            "days_in_pipeline": 60,
        },
    ],
    "fraud_detection": [
        {
            "amount": 4200.0,
            "hour_of_day": 2,
            "merchant_category": "Electronics",
            "distance_from_home_km": 850.0,
            "chip_used": "No",
            "transaction_count_24h": 8,
        },
        {
            "amount": 45.0,
            "hour_of_day": 14,
            "merchant_category": "Groceries",
            "distance_from_home_km": 3.0,
            "chip_used": "Yes",
            "transaction_count_24h": 1,
        },
    ],
    "loan_default": [
        {
            "credit_score": 580,
            "annual_income": 42000,
            "loan_amount": 35000,
            "employment_years": 1,
            "debt_to_income": 0.48,
            "loan_purpose": "Personal",
        },
        {
            "credit_score": 740,
            "annual_income": 95000,
            "loan_amount": 20000,
            "employment_years": 8,
            "debt_to_income": 0.22,
            "loan_purpose": "Home improvement",
        },
    ],
}

DEMO_SAMPLE_ROWS: dict[str, dict] = {
    "customer_churn": SAMPLE_BATCHES["customer_churn"][0],
    "lead_scoring": SAMPLE_BATCHES["lead_scoring"][0],
    "fraud_detection": SAMPLE_BATCHES["fraud_detection"][0],
    "loan_default": SAMPLE_BATCHES["loan_default"][0],
}


def list_use_cases() -> list[dict]:
    return [
        {
            "id": uc["id"],
            "name": uc["name"],
            "industry": uc["industry"],
            "description": uc["description"],
            "business_goal": uc["business_goal"],
            "target_col": uc["target_col"],
            "problem_type": uc["problem_type"],
        }
        for uc in USE_CASES.values()
    ]


def get_use_case(use_case_id: str) -> dict | None:
    return USE_CASES.get(use_case_id)


def load_demo_dataset(use_case_id: str, upload_dir: str = "uploads") -> dict:
    """Copy bundled demo CSV into uploads/ and return upload metadata."""
    uc = USE_CASES.get(use_case_id)
    if not uc:
        raise ValueError(f"Unknown use case: {use_case_id}")

    src = uc["dataset_file"]
    if not os.path.exists(src):
        raise FileNotFoundError(f"Demo dataset missing: {src}")

    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.basename(src)
    dest = os.path.join(upload_dir, filename)
    shutil.copy2(src, dest)

    return {
        "use_case_id": use_case_id,
        "filename": filename,
        "filepath": dest,
        "target_col": uc["target_col"],
        "use_case": uc,
    }


def risk_tier(probability: float, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {"high": 0.65, "medium": 0.35}
    if probability >= thresholds["high"]:
        return "high"
    if probability >= thresholds["medium"]:
        return "medium"
    return "low"


def recommended_action(use_case: dict | None, tier: str) -> str:
    if not use_case:
        return "Review prediction with domain expert."
    return use_case.get("actions", {}).get(tier, "No action defined.")


def get_sample_batch(use_case_id: str) -> list[dict]:
    return SAMPLE_BATCHES.get(use_case_id, [])


def get_demo_sample_row(use_case_id: str) -> dict | None:
    return DEMO_SAMPLE_ROWS.get(use_case_id)
