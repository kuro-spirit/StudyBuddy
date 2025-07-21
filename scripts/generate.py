from llama_cpp import Llama
from retrieve import retrieve_top_k
import os
from datetime import datetime

llm = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", 
    n_gpu_layers=-1, # Gpu layers set to 32 for partial offload
    n_ctx=8192,   # Increase context window to avoid overflow
    n_threads=os.cpu_count(),  # Adjust to your CPU core count
    verbose=False
)

def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are an expert study assistant helping an university student understand course material. 
    Only use the provided context below to answer the question.
    If technical terms appear, briefly define them.
    If the answer cannot be found in the context, respond with "I cannot find any context in your
    notes". It is okay to not find any answer in the context.

    Context:
    {context}

    Question: {query}
    Answer:"""
    prompt2 = f"""<s>[INST] 
    You are a helpful study assistant. Use only the context provided below to answer the user's question.
    If the answer is not in the context, say "I cannot find any context in your notes."

    Context:
    {context}

    Question: {query}
    [/INST]"""
    # print(f"\n[DEBUG] Prompt Sent to LLaMA:\n{prompt[:1000]}...\n")
    return prompt2

def answer_question(query: str) -> str:
    # Retrieve relevant context
    chunks = retrieve_top_k(query)

    # Build prompt
    prompt = build_prompt(query, chunks)

    # Generate response
    response = llm(prompt, max_tokens=512, stop=["\n"])
    print("\n[DEBUG] Raw LLaMA response:\n", response)

    # clean response
    answer = response["choices"][0].get("text", "").strip()
    print("\n[DEBUG] Cleaned response:\n", answer)
    return answer

if __name__ == "__main__":
    asking = True
    while asking:
        question = input("Write your question: ")
        start_time = datetime.now()
        answer = answer_question(question)
        end_time = datetime.now()
        print("\n--- Answer ---\n")
        print(answer)
        print(f"Time to response: {end_time - start_time}")
        cont = input("Continue asking? input y for Yes and n for No: ")
        if cont == "n":
            asking = not asking
        