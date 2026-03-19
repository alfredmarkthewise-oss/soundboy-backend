import base64
import requests
import json
import os

# 1. Encode the Spectrogram Image
# We have to convert the saved image into a base64 string so it can travel via API
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "ai_hearing_output.png"
base64_image = encode_image(image_path)

# 2. Setup the API Connection (Using OpenAI as the example framework)
api_key = os.environ.get("OPENAI_API_KEY") # Securely load from your environment

if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. "
                       "Please set it with: export OPENAI_API_KEY='your-key-here'")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 3. Design the Agent's Prompt (The Engineering Logic)
# We embed the acoustic principles directly into the system prompt.
system_prompt = """
You are an expert audio engineer and DSP analyst.
You are looking at a Mel-spectrogram of a raw audio recording.
Your philosophy is: "Dynamic range is beauty, that's where intent lives."
Analyze this visual data and provide recommendations based on:
1. Subtractive EQ to remove disharmonic frequencies.
2. Compression settings to manage dynamics and enhance harmonics.
3. Time and Space effects (Reverb and Delay) if the signal lacks presence.

You must return ONLY a valid JSON object. Do not include markdown formatting like ```json.
"""

user_prompt = "Analyze this spectrogram and provide a mixing JSON."

# 4. Construct the Payload
payload = {
    "model": "gpt-4o", # A multimodal model capable of vision
    "messages": [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high" # Forces the AI to look closely at the frequency grid
                    }
                }
            ]
        }
    ],
    "max_tokens": 500,
    "temperature": 0.2 # Low temperature so the math/suggestions remain objective and strict
}

# 5. Execute the Agentic Call
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
data = response.json()

# Extract and parse the JSON string returned by the AI
try:
    ai_mixing_decisions = json.loads(data['choices'][0]['message']['content'])
    print(json.dumps(ai_mixing_decisions, indent=4))
except KeyError as e:
    print(f"API Error: {data}")
