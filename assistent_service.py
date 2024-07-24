from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Initialize Flask application
app = Flask(__name__)

# Model path
model_path = '/root/autodl-tmp/qwen/Qwen2-7B-Instruct'

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Conversation history
history = []

def format_history(history, prompt):
    messages = [{"role": "user", "content": msg} for msg in history]
    messages.append({"role": "user", "content": prompt})
    return messages

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 256)

    # Format input
    messages = format_history(history, prompt)
    formatted_input = " ".join([msg["content"] for msg in messages])

    # Generate response
    response = pipe(formatted_input, max_length=max_length, truncation=True)
    assistant_reply = response[0]['generated_text']

    # Update history
    history.append(prompt)
    history.append(assistant_reply)

    return jsonify({"response": assistant_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
