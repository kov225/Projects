import json
import re
import os

def parse_image_number_ai(img_str):
    """Extracts integer from AI image string (e.g., 'IMG_0038' -> 38)."""
    if not img_str:
        return -1
    match = re.search(r'IMG_(\d+)', img_str)
    if match:
        return int(match.group(1))
    return -1

def flatten_ai_defendants(defendants_list):
    """
    Flattens the structured AI defendant list into a list of strings 
    to match the unstructured Human GT format.
    """
    flat_list = []
    for d in defendants_list:
        # Name components
        if d.get("lastName"): flat_list.append(d["lastName"])
        if d.get("firstName"): flat_list.append(d["firstName"])
        
        # Location (IF AVAILABLE)
        if d.get("location"): 
            flat_list.append("of " + d["location"])

        # County (IF AVAILABLE)
        if d.get("county"): 
            flat_list.append(d["county"])

        # Occupations
        if d.get("occupations"):
            flat_list.extend([o for o in d["occupations"] if o])
            
      #   # Description/Extra
      #   if d.get("description"):
      #       flat_list.append(d["description"])
            
    return flat_list

def flatten_ai_plaintiffs(plaintiffs_list):
    """Flattens AI plaintiffs."""
    flat_list = []
    for p in plaintiffs_list:
        if p.get("entityName"):
            flat_list.append(p["entityName"])
        if p.get("lastName"): flat_list.append(p["lastName"])
        if p.get("firstName"): flat_list.append(p["firstName"])
        
    return flat_list

def build_places_for_ai_case(case):
    """
    Builds the 'places' list from defendant locations/counties.
    Fallback to top-level county if no defendant locations found.
    """
    places = []
    has_defendant_location = False
    
    for d in case.get("defendants", []):
        if d.get("location"):
            places.append(d["location"])
            has_defendant_location = True
        if d.get("county"):
            places.append(d["county"])
            has_defendant_location = True
            
    if not has_defendant_location and case.get("county"):
        places.append(case.get("county"))
        
    return places

def load_and_convert():
    standardized_data = []

    # Load AI HTR Data
    print("Loading AI HTR Dataset...")
    try:
        with open('datasets/AI_HTR_Output.json', 'r') as f:
            ai_json = json.load(f)
            # AI data is wrapped in "cases" list
            ai_cases = ai_json.get("cases", [])
        
        for idx, case in enumerate(ai_cases):
            
            # AI uses 'source_image_directory' like 'IMG_0038'
            img_dir = case.get("source_image_directory", "")
            
            plaintiffs_flat = flatten_ai_plaintiffs(case.get("plaintiffs", []))
            defendants_flat = flatten_ai_defendants(case.get("defendants", []))
            places = build_places_for_ai_case(case)
            
            
            # For each case in ai_cases, if the object property "isCrownPlaintiff" is true, then append a default plaintiff "Rex" to the plaintiffs_flat list
            if case.get("isCrownPlaintiff", True) and len(plaintiffs_flat) == 0:
                plaintiffs_flat.append("Rex")

            plea = ""
            plea_details = ""

            # Construct full text for similarity matching later
            full_text_parts = plaintiffs_flat + defendants_flat + places
            if "plea" in case:
                plea = case["plea"].get("primary_charge", "")
                if case["plea"].get("details"):
                    plea_details = case["plea"]["details"]
                    full_text_parts.append(case["plea"]["details"])
            
            std_case = {
                "ai_caseid": idx+1,
                "source": "ai_htr",
                "image_num": parse_image_number_ai(img_dir),
                "original_image_id": img_dir,
                "county": case.get("county", ""),
                "places": places,
                "plaintiffs": plaintiffs_flat,
                "defendants": defendants_flat,
                "plea": plea,
                "plea_details": plea_details,
                "full_text": " ".join(full_text_parts)
            }
            standardized_data.append(std_case)
            
    except FileNotFoundError:
        print("Error: datasets/AI_HTR_Output.json not found.")
        
    return standardized_data

if __name__ == "__main__":
    data = load_and_convert()
    print(f"Successfully processed {len(data)} records.")

    # Save to file
    output_path = 'datasets/standardized_ai_data.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved standardized data to '{output_path}'")

