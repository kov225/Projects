from flask import Flask, render_template, jsonify, request
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask with custom template folder if needed
# Since app.py is in web/, it will look for web/templates/ by default
app = Flask(__name__, template_folder='templates')

# Paths relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MASTER_RECON_PATH = DATA_DIR / "master_reconciliation.json"
ACCURACY_PATH = DATA_DIR / "FINAL_ACCURACY.json"

def load_json(path):
    if not path.exists():
        logger.warning(f"JSON file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/")
def index():
    accuracy = load_json(ACCURACY_PATH)
    master = load_json(MASTER_RECON_PATH) or []
    
    # Get top cases (ranked by F1)
    sorted_cases = sorted(
        master, 
        key=lambda x: x.get("accuracy_metrics_v1", {}).get("name_f1", 0), 
        reverse=True
    )
    
    return render_template(
        "index.html", 
        summary=accuracy.get("summary", {}),
        top_cases=sorted_cases[:5]
    )

@app.route("/reconciliation")
def reconciliation():
    master = load_json(MASTER_RECON_PATH) or []
    page = int(request.args.get('page', 1))
    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    
    return render_template(
        "reconciliation.html", 
        cases=master[start:end],
        page=page,
        total_pages=(len(master) // per_page) + 1
    )

@app.route("/case/<int:case_id>")
def case_detail(case_id):
    master = load_json(MASTER_RECON_PATH) or []
    case = next((c for c in master if c.get("human_CaseId") == case_id), None)
    if not case:
        return "Case not found", 404
    
    return render_template("case_detail.html", case=case)

@app.route("/api/stats")
def stats():
    accuracy = load_json(ACCURACY_PATH)
    return jsonify(accuracy.get("summary", {}))

if __name__ == "__main__":
    logger.info("Starting professional dashboard...")
    app.run(debug=True, port=5000)
