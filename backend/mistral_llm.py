from llama_cpp import Llama

llm = Llama(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=4096, n_threads=8)

def generate_answer(prompt):
    output = llm(prompt, max_tokens=500)
    return output['choices'][0]['text'].strip()

