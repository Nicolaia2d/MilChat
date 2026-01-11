import requests

url = "http://127.0.0.1:5000/ask"
data = {"question": "Hva er rekkevidden til AG3?"}

response = requests.post(url, json=data)

if response.ok:
    print("✅ Svar mottatt:")
    print(response.json())
else:
    print("❌ Noe gikk galt:")
    print(response.status_code)
    print(response.text)
