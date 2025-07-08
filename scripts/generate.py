from llama_cpp import Llama
from retrieve import retrieve_top_k

llm = Llama(
    model_path="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",  # or full path if needed
    n_ctx=4096,   # Increase context window to avoid overflow
    n_threads=4,  # Adjust to your CPU core count
    verbose=True  # Optional, shows model loading info
)

def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful study assistant.

    Context:
    {context}

    Question: {query}
    Answer:"""
    return prompt

def answer_question(query: str, k: int = 5) -> str:
    # Retrieve relevant context
    chunks = retrieve_top_k(query, k=k)

    # Build prompt
    prompt = build_prompt(query, chunks)

    # Generate response
    response = llm(prompt, max_tokens=512, stop=["\n", "User:"])
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    question = "What is a rank"
    answer = answer_question(question)
    print("\n--- Answer ---\n")
    print(answer)