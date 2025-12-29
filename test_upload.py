import requests

# URL of your ingest server
url = 'http://localhost:8000/upload-data'

# The files you want to upload (ensure these exist in your folder!)
# Update filenames if yours are different
files = {
    'attribution_file': open(r'C:\Users\Krina\atlas\data\ga_2.csv', 'rb'),
    'mmm_file': open(r'C:\Users\Krina\atlas\data\mmm_data_2016_2017.csv', 'rb')
}

print("📤 Sending files to Ingest API...")
try:
    response = requests.post(url, files=files)
    print(f"✅ Status Code: {response.status_code}")
    print(f"📜 Response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")