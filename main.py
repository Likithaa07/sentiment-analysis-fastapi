
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import threading

# Enable nested event loops for Colab async compatibility
nest_asyncio.apply()

app = FastAPI()

# Load your sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class TextRequest(BaseModel):
    text: str

# API endpoint for sentiment prediction
@app.post("/predict")
def predict(request: TextRequest):
    result = classifier(request.text)
    return {"label": result[0]['label'], "score": result[0]['score']}

# Serve your frontend HTML page
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sentiment Analysis Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; max-width: 600px; }
        textarea { width: 100%; height: 100px; font-size: 1rem; }
        button { margin-top: 10px; padding: 8px 16px; font-size: 1rem; }
        #result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h2>Sentiment Analysis</h2>
    <textarea id="inputText" placeholder="Type your text here..."></textarea><br />
    <button onclick="getSentiment()">Analyze Sentiment</button>
    <div id="result"></div>
<script>
async function getSentiment() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) {
        alert('Please enter some text!');
        return;
    }
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = "Analyzing...";
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) {
            throw new Error('Error from API: ' + response.status);
        }
        const data = await response.json();
        resultDiv.textContent = `Sentiment: ${data.label} (Confidence: ${(data.score * 100).toFixed(2)}%)`;
    } catch (error) {
        resultDiv.textContent = 'Error: ' + error.message;
    }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return html_content

# Start ngrok tunnel to expose port 8000
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run Uvicorn server in a separate thread so it doesn't block Colab
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run)
thread.start()
