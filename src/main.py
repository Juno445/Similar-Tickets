# === Imports ===
from urllib.parse import quote
import requests
from requests.auth import HTTPBasicAuth
import json
import csv
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import logging
import smtplib
from email.message import EmailMessage

# === Configuration ===
API_KEY = "X" # Replace with your Freshservice API key
DOMAIN = "X" # Replace with your Freshservice domain (e.g., "yourcompany")
DUPLICATE_CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'similarity_threshold': 0.87,
    'filter_by_requester': True,
    'max_date_diff_days': None,  # Optional: set to None to disable
    'subject_weight': 0.2      # 0.3 = 30% subject, 70% description
}

# === Utility Functions ===
def sanitize_html(html_string):
    if html_string is None:
        return ""
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text(strip=True)

# --------------------- Ticket Fetching ---------------------
def fetch_dispatch_tickets(csv_filename):
    query = "(status:2 OR status:3 OR status:9 OR status:12 OR status:18 OR status:21) AND (group_id:X OR group_id:X)" # Replace with your group IDs
    encoded_query = quote(query)
    # Base URL
    base_url = f"https://{DOMAIN}.freshservice.com/api/v2/tickets/filter?query=\"{encoded_query}\""
    # Parameters for pagination
    page = 1
    per_page = 100  # Max allowed per page (may vary by API version)
    all_tickets = []
    while True:
        url = f"{base_url}&page={page}&per_page={per_page}"
        resp = requests.get(url, auth=HTTPBasicAuth(API_KEY, 'X'))
        if resp.status_code != 200:
            print(f"Error: HTTP {resp.status_code}\n{resp.text}")
            break
        data = resp.json()
        tickets = data.get("tickets", [])
        all_tickets.extend(tickets)
        if not tickets:
            break
        page += 1
    slim_tickets = [
        {
            "Ticket ID": t.get("id"),
            "Subject": t.get("subject"),
            "Requester Email": t.get("requester_id"),
            "Date Created": t.get("created_at"),
            "Description": sanitize_html(t.get("description"))
        }
        for t in all_tickets
    ]
    print(json.dumps(slim_tickets, indent=2))
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Ticket ID', 'Subject', 'Requester Email', 'Date Created', 'Description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(slim_tickets)
    return slim_tickets

# --------------------- Duplicate Detection ---------------------
def load_and_prepare_data(input_file):
    df = pd.read_csv(input_file)
    required_cols = [
        'Ticket ID', 'Subject', 'Requester Email', 'Date Created', 'Description'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=['Subject', 'Description'])
    return df

def find_duplicates(df, CONFIG):
    model = SentenceTransformer(CONFIG['embedding_model'])
    alpha = CONFIG.get('subject_weight', 0.3)  # weight for subject (0.0-1.0)
    
    # Embedding subject and description separately
    subjects = df['Subject'].fillna("").tolist()
    descriptions = df['Description'].fillna("").tolist()
    
    subject_emb = model.encode(subjects, convert_to_tensor=True, show_progress_bar=True)
    desc_emb = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    
    # Weighted combination
    embeddings = alpha * subject_emb + (1 - alpha) * desc_emb
    
    pairs = []
    total = len(df)
    for i in tqdm(range(total), desc="Finding duplicates"):
        for j in range(i + 1, total):
            sim_score = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim_score < CONFIG['similarity_threshold']:
                continue
            row_i = df.iloc[i]
            row_j = df.iloc[j]
            if CONFIG['filter_by_requester'] and row_i['Requester Email'] != row_j['Requester Email']:
                continue
            if CONFIG['max_date_diff_days']:
                date1 = pd.to_datetime(row_i['Date Created'])
                date2 = pd.to_datetime(row_j['Date Created'])
                if abs((date1 - date2).days) > CONFIG['max_date_diff_days']:
                    continue
            pair = {
                'Ticket ID 1': row_i['Ticket ID'],
                'Subject 1': row_i['Subject'],
                'Requester Email 1': row_i['Requester Email'],
                'Date Created 1': row_i['Date Created'],
                'Description 1': row_i['Description'],
                'Ticket ID 2': row_j['Ticket ID'],
                'Subject 2': row_j['Subject'],
                'Requester Email 2': row_j['Requester Email'],
                'Date Created 2': row_j['Date Created'],
                'Description 2': row_j['Description'],
                'Similarity Score': sim_score
            }
            pairs.append(pair)
    return pd.DataFrame(pairs)

def save_duplicates(df_duplicates, output_file):
    if not df_duplicates.empty:
        df_duplicates.to_csv(output_file, index=False)
        logging.info(f"Saved {len(df_duplicates)} potential duplicates to {output_file}")
    else:
        logging.info("No potential duplicates found.")

# --------------------- Master Main ---------------------
def main():
    logging.basicConfig(level=logging.INFO)
    # No dialogs, fixed filenames:
    tickets_csv = "data/tickets.csv"
    output_csv = "data/potential_duplicates.csv"
    # Step 1: Fetch tickets and save to tickets.csv
    logging.info("Fetching tickets...")
    tickets = fetch_dispatch_tickets(tickets_csv)
    if not tickets:
        logging.error("No tickets found or fetch failed.")
        return
    logging.info(f"Successfully fetched {len(tickets)} tickets.")
    # Step 2: Detect and save duplicates
    logging.info("Running duplicate detection...")
    df = load_and_prepare_data(tickets_csv)
    duplicates = find_duplicates(df, DUPLICATE_CONFIG)
    save_duplicates(duplicates, output_csv)

if __name__ == "__main__":
    main()