import json
import re
from bs4 import BeautifulSoup

def parse_kb_html(file_path):
    # 1. Read the HTML file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # 2. Initialize BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # 3. Find all table rows
    # Note: We find all 'tr' tags. We will filter out the header row inside the loop.
    rows = soup.find_all('tr')

    json_output = []
    row_num = 1

    for row in rows:
        cells = row.find_all('td')

        # Skip rows that don't have enough columns (e.g., empty rows)
        if len(cells) < 5:
            continue

        # Get raw text from cells, stripping whitespace
        # cell 0 = Image, 1 = County, 2 = Plaintiffs, 3 = Defendants, 4 = Plea
        c_image = cells[0].get_text(strip=True)
        c_county = cells[1].get_text(strip=True)
        c_plaintiffs = cells[2].get_text(strip=True)
        c_defendants = cells[3].get_text(strip=True)
        c_plea = cells[4].get_text(strip=True)

        # Skip the Header Row
        # We check if the first column contains the word "Image" (case-insensitive)
        if "Image" in c_image:
            continue

        # --- LOGIC 1: Extract Image Number ---
        # Extracts the first sequence of digits found in the string (e.g., "f 9" -> 9)
        img_num_match = re.search(r'\d+', c_image)
        image_num = int(img_num_match.group()) if img_num_match else None

        # --- LOGIC 2: Split Plaintiffs and Defendants ---
        # regex split on comma (,) OR semicolon (;)
        # list comprehension strips whitespace and removes empty strings
        
        plaintiffs_list = [
            name.strip() 
            for name in re.split(r'[,;]', c_plaintiffs) 
            if name.strip()
        ]
        
        defendants_list = [
            name.strip() 
            for name in re.split(r'[,;]', c_defendants) 
            if name.strip()
        ]

        # 4. Construct the Data Object
        entry = {
            "human_case_id": row_num,
            "image_id": c_image,
            "image_num": image_num,
            "county": c_county,
            "plaintiffs": plaintiffs_list,
            "defendants": defendants_list,
            "plea": c_plea
        }
        
        row_num += 1

        json_output.append(entry)

    return json_output

# --- Execution ---
if __name__ == "__main__":
    input_filename = "datasets/KB_Table.html"
    output_filename = "datasets/KB_Table_Parsed.json"

    print(f"Parsing {input_filename}...")
    data = parse_kb_html(input_filename)

    if data:
        # Save to JSON file
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully processed {len(data)} rows.")
        print(f"Data saved to: {output_filename}")
        
        # Optional: Print the first item to verify structure matches your requirements
        print("\n--- Preview of first item ---")
        print(json.dumps(data[0], indent=4))