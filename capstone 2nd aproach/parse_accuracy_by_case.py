import json
import os

def calculate_final_accuracy(input_path="datasets/master_reconciliation.json", output_path="datasets/FINAL_ACCURACY.json"):
    """
    Reads master_reconciliation.json and generates a summarized accuracy report.
    """
    
    print(f"Loading reconciliation data from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_cases = len(data)
    if total_cases == 0:
        print("No cases found in input file.")
        return

    # Initialize aggregators
    county_matches = 0
    plea_matches = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    # List for case-by-case details
    case_details = []
    
    for entry in data:
        # Get Human Case ID
        human_id = entry.get('human_CaseId')
        
        # Get Metrics
        metrics = entry.get('accuracy_metrics_v1', {})
        
        # Update Aggregates
        if metrics.get('county_match'): county_matches += 1
        if metrics.get('plea_match'): plea_matches += 1
        total_precision += metrics.get('name_precision', 0.0)
        total_recall += metrics.get('name_recall', 0.0)
        total_f1 += metrics.get('name_f1', 0.0)
        
        # Add to detailed list
        # Using simple dict format: { "caseId": X, "metrics": {...} }
        case_details.append({
            "case_id": human_id,
            "metrics": {
                "county_match": metrics.get('county_match'),
                "plea_match": metrics.get('plea_match'),
                "name_precision": metrics.get('name_precision'),
                "name_recall": metrics.get('name_recall'),
                "name_f1": metrics.get('name_f1')
            }
        })
        
    # Calculate Averages
    avg_precision = round(total_precision / total_cases, 4)
    avg_recall = round(total_recall / total_cases, 4)
    avg_f1 = round(total_f1 / total_cases, 4)
    county_accuracy = round((county_matches / total_cases) * 100, 2)
    plea_accuracy = round((plea_matches / total_cases) * 100, 2)
    
    # Construct Final JSON Object
    final_output = {
        "summary": {
            "total_cases": total_cases,
            "county_match_accuracy_percent": county_accuracy,
            "plea_match_accuracy_percent": plea_accuracy,
            "average_name_precision": avg_precision,
            "average_name_recall": avg_recall,
            "average_name_f1_score": avg_f1
        },
        "case_accuracy_details": case_details
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Successfully generated {output_path}")
    print("Summary:")
    print(json.dumps(final_output["summary"], indent=2))

if __name__ == "__main__":
    calculate_final_accuracy()



