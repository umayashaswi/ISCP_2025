# ISCP 2025 Flipkart CTF Challenge 

This repository contains the solution for the CTF Challenge. The goal is to **detect and redact Personally Identifiable Information (PII)** from the provided dataset.

## 📂 Repository Contents

- `detector_full_candidate_name.py` → Python script to detect and redact PII.  
- `iscp_pii_dataset_-_Sheet1.csv` → Input dataset for the challenge.  
- `redacted_output_candidate_full_name.csv` → Output file with PII redacted.  

## ⚙️ Setup & Usage

```bash
git clone https://github.com/ManchineellaLikhitha/SOC.git
cd SOC
python detector_full_candidate_name.py iscp_pii_dataset_-_Sheet1.csv
