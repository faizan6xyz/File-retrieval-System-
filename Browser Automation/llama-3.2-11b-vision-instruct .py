from openai import OpenAI
import base64

# Step 2: Encode to base64
with open(r"dataset/001_github_com_faizan6xyz.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

image_url = f"data:image/png;base64,{encoded}"

# Step 3: Send to NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

completion = client.chat.completions.create(
    model="meta/llama-3.2-11b-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "List all buttons and their position coordinates."}
        ]
    }],
    max_tokens=1024,
    stream=False
)

print(completion.choices[0].message.content)