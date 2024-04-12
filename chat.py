from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re

model_id = "google/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
)


response_pattern = r"<start_of_turn>model\n(.*?)<eos>"
cond = True

while cond:
    user_prompt = input("You: ")
    if user_prompt == "EXIT":
        cond = False
        break
    chat = [
        { "role": "user", "content": user_prompt },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=256)

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # TODO: extract user feedback from response

    matches = re.search(response_pattern, response)
    if matches:
        print(f"Gemma: {matches.group(1)}")
    else:
        print("Invalid response.")


    # TODO: update LLM with user feedback









