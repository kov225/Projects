import pytest
import json
from src.engine import KingsBenchReconciler, group_tokens_into_entities, calculate_accuracy_metrics

def test_group_tokens():
    tokens = ["John", "Doe", "yeoman", "of", "London"]
    entities = group_tokens_into_entities(tokens)
    assert "Doe John" in entities
    assert "London" not in entities # Because 'of' prefix skips it

def test_accuracy_metrics_identity():
    case_a = {
        "county": "Surrey",
        "plea": "Trespass",
        "plaintiffs": ["John Smith"],
        "defendants": ["Jane Doe"]
    }
    metrics = calculate_accuracy_metrics(case_a, case_a)
    assert metrics["county_match"] is True
    assert metrics["name_f1"] == 1.0

def test_reconciler_simple_match():
    gt = [{
        "human_case_id": 1,
        "image_num": 10,
        "county": "Kent",
        "plaintiffs": ["Adam"],
        "defendants": ["Eve"],
        "plea": "Debt"
    }]
    ai = [{
        "ai_caseid": 101,
        "image_num": 10,
        "county": "Kent",
        "plaintiffs": ["Adam"],
        "defendants": ["Eve"],
        "plea": "Debt",
        "full_text": "Adam Eve Kent Debt"
    }]
    
    reconciler = KingsBenchReconciler(gt, ai)
    splits, master = reconciler.run_reconciliation()
    
    assert len(master) == 1
    assert master[0]["reconciled_ai_data"]["reconciled_id"] == 101
    assert master[0]["accuracy_metrics_v1"]["name_f1"] > 0.9

if __name__ == "__main__":
    pytest.main([__file__])
