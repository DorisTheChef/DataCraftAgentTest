import os
import google.generativeai as genai
import pandas as pd
import json
import re

# Read Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

def generate_patients_with_gemini(n=100):
    prompt = (
        f"Generate {n} sets of virtual patient data in JSON array format. "
        "Each patient should have: Name, Age, Gender, and Condition. "
        "Example: "
        '[{"Name": "John Smith", "Age": 45, "Gender": "Male", "Condition": "Diabetes"}, ...]'
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    # Extract JSON from response
    match = re.search(r'\[.*\]', response.text, re.DOTALL)
    if match:
        patients = json.loads(match.group(0))
        return patients
    else:
        raise ValueError("Could not find JSON array in Gemini response.")

if __name__ == "__main__":
    patients = generate_patients_with_gemini(100)
    df = pd.DataFrame(patients)
    df.to_csv("gemini_virtual_patients.csv", index=False)
    print("Generated 100 virtual patients using Gemini API and saved to gemini_virtual_patients.csv")