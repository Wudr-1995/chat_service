from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from colorama import init, Fore, Style
import torch

# Load model and tokenizer
model_path = '/root/autodl-tmp/qwen/Qwen2-7B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to format the history and current prompt for the model
def format_history(history, prompt):
    messages = [{"role": "user", "history_content": msg} for msg in history]
    messages.append({"role": "user", "content": prompt})
    return messages

# Function to generate and stream output
def generate_streamed_response(model, tokenizer, input_text, max_new_tokens=None):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)  # Create attention mask

    # Set initial values
    output_ids = input_ids
    stop_generation = False
    max_iterations = 1000
    iteration = 0

    stop_tokens = ['Human', 'assistant']

    while not stop_generation and iteration < max_iterations:
        output = model.generate(output_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True)
        new_tokens = output[0][output_ids.shape[-1]:]  # Get new tokens generated
        decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

        for st in stop_tokens:
            if st in decoded_output:
                stop_generation = True

        if stop_generation:
            break

        for char in decoded_output:
            if char == '\n':
                print()  # Print a newline character
            else:
                print(char, end='', flush=True)  # Print each character

        output_ids = torch.cat((output_ids, new_tokens.unsqueeze(0)), dim=-1)
        attention_mask = torch.ones(output_ids.shape, device=device)

        iteration += 1


# Start interactive dialogue
print("Start chatting with the model (type 'exit' to stop):")
history = []
while True:
    # Get user input
    user_input = input(Fore.RED + "User: " + Style.RESET_ALL)
    print()
    if user_input.lower() == 'exit':
        break

    # Format input with history
    messages = format_history(history, user_input)
    formatted_input = " ".join([msg["content"] for msg in messages])

    # Generate and stream response
    print(Fore.BLUE + 'Assistant: ' + Style.RESET_ALL, end='', flush=True)
    generate_streamed_response(model, tokenizer, formatted_input, max_new_tokens=10)
    print()

    # Update history
    history.append(user_input)
    history.append(formatted_input)

print("Chat ended.")
