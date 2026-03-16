"""
llm_explanation.py

Generates regulatory-style adverse action notices using either GPT-4o-mini
(OpenAI) or Mistral-7B (Ollama), controlled by the EXPLANATION_BACKEND env var.

The adverse action notice is a real compliance artifact. Under ECOA and FCRA,
a lender must provide specific reasons for denying credit within 30 days. The
reasons must be the actual factors that most significantly influenced the
decision, which is exactly what SHAP values give us.

The prompt is carefully engineered to produce notices that match the format and
language regulators expect. Framing this as a legal compliance feature rather
than a chatbot gimmick makes it genuinely differentiated on a resume.
"""

import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ADVERSE_ACTION_PROMPT = """You are a compliance officer at a consumer lending institution.
Write a formal adverse action notice as required by the Equal Credit Opportunity Act (ECOA) 
and the Fair Credit Reporting Act (FCRA).

The applicant's loan application was denied. The following factors, identified by a 
machine learning risk model, most significantly influenced the denial decision:

{shap_summary}

Additional context:
- Applicant risk score: {risk_score:.4f} (threshold for approval: {threshold:.4f})
- Model version: {model_version}

Write a professional adverse action notice that:
1. States the credit decision clearly in the first sentence
2. Lists the specific reasons in order of significance (use the factor names provided)
3. Informs the applicant of their right to a free credit report within 60 days
4. Informs the applicant of their right to dispute inaccurate information with the credit bureau
5. Uses plain language that a general audience can understand
6. Does NOT mention the machine learning model or SHAP values by name
7. Is between 150 and 250 words

Write only the notice text, no preamble or explanation."""


def format_shap_for_prompt(shap_dict: dict[str, float]) -> str:
    """Convert the SHAP dict to a readable factor list for the prompt."""
    lines = []
    for i, (feature, value) in enumerate(shap_dict.items(), 1):
        direction = "negatively" if value > 0 else "positively"
        # Translate internal feature names to human-readable descriptions
        readable = _feature_to_readable(feature)
        lines.append(f"{i}. {readable} ({direction} impacted the decision)")
    return "\n".join(lines)


def _feature_to_readable(feature_name: str) -> str:
    """Map internal feature names to ECOA-compliant plain language descriptions."""
    mappings = {
        "dti": "High debt-to-income ratio",
        "fico_score": "Credit score below required threshold",
        "fico_bucket": "Credit score classification",
        "delinq_2yrs": "Recent delinquencies on credit accounts",
        "revol_util": "High revolving credit utilization",
        "annual_inc": "Insufficient annual income",
        "debt_burden_ratio": "High monthly debt burden relative to income",
        "dti_x_unemployment": "Elevated debt burden given current economic conditions",
        "employment_stability": "Limited employment history",
        "emp_length": "Insufficient length of employment",
        "pub_rec": "Public records on file (bankruptcies or judgments)",
        "open_acc": "Number of open credit accounts",
        "loan_amnt": "Requested loan amount",
        "int_rate": "Interest rate on existing obligations",
        "unemployment_rate": "Current macroeconomic conditions",
        "grade_enc": "Overall credit grade assessment",
    }
    # fall back to splitting underscores for unmapped features
    return mappings.get(feature_name, feature_name.replace("_", " ").title())


def generate_openai(prompt: str) -> str:
    """Call GPT-4o-mini via the OpenAI Python client."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,   # low temperature for consistent regulatory language
    )
    return resp.choices[0].message.content.strip()


def generate_ollama(prompt: str) -> str:
    """Call Mistral-7B via Ollama's REST API.
    
    Ollama runs locally and the endpoint is configured via OLLAMA_BASE_URL.
    This is the zero-cost path and also makes for a stronger resume story
    because it directly references the Mistral optimization work.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base_url}/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 400},
            },
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


def generate_adverse_action_notice(
    shap_values: dict[str, float],
    risk_score: float,
    decision_threshold: float,
    model_version: str,
) -> str:
    """Generate a regulatory-style adverse action notice from SHAP values.
    
    Dispatches to OpenAI or Ollama based on EXPLANATION_BACKEND config.
    Falls back to a template-based notice if the LLM call fails, because
    a serving endpoint that errors on /explain would block compliance workflows.
    """
    backend = os.environ.get("EXPLANATION_BACKEND", "ollama").lower()
    shap_summary = format_shap_for_prompt(shap_values)

    prompt = ADVERSE_ACTION_PROMPT.format(
        shap_summary=shap_summary,
        risk_score=risk_score,
        threshold=decision_threshold,
        model_version=model_version,
    )

    try:
        if backend == "openai":
            notice = generate_openai(prompt)
        elif backend == "ollama":
            notice = generate_ollama(prompt)
        else:
            raise ValueError(f"Unknown EXPLANATION_BACKEND: '{backend}'. Use 'openai' or 'ollama'.")

        logger.info(f"Adverse action notice generated via {backend} ({len(notice)} chars)")
        return notice

    except Exception as e:
        logger.error(f"LLM explanation failed ({backend}): {e}")
        return _fallback_notice(shap_values, risk_score)


def _fallback_notice(shap_values: dict[str, float], risk_score: float) -> str:
    """Template-based fallback notice when the LLM is unavailable."""
    reasons = [_feature_to_readable(f) for f in list(shap_values.keys())[:3]]
    reasons_text = "; ".join(reasons)
    return (
        f"We regret to inform you that your credit application has been denied. "
        f"The primary reasons for this decision include: {reasons_text}. "
        f"You have the right to receive a free copy of your credit report from "
        f"the consumer reporting agency used in this evaluation within 60 days. "
        f"You also have the right to dispute any inaccurate information in your "
        f"credit report by contacting the agency directly."
    )
