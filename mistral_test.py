from ollama import chat

response = chat(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "What is fuel?"
        }
    ]
)

print(response["message"]["content"])