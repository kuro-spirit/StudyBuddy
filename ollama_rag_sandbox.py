# ollama_rag_sandbox.py

# MODIFIED ollama_rag_sandbox.py (only change the build_llama2_rag_prompt function)

import ollama

def build_llama2_rag_prompt(user_question: str, retrieved_context: list[str]) -> list[dict]:
    """
    Constructs a Llama 2 compatible RAG prompt as a list of message dictionaries.
    Optimized for better context adherence and direct answering.

    Args:
        user_question: The question posed by the user.
        retrieved_context: A list of relevant text chunks (strings) to act as context.

    Returns:
        A list of dictionaries formatted for Ollama's chat API,
        including a system message and a user message with context.
    """

    # --- 1. System Prompt (No Change - still good) ---
    system_message = (
        "You are a highly knowledgeable and precise AI assistant. "
        "Your primary goal is to answer questions *ONLY* based on the provided 'Context from study materials'. "
        "If the answer is not explicitly present in the provided context, "
        "you MUST state 'I cannot find enough information in the provided materials to answer this question.' "
        "Do NOT use any outside knowledge or make up facts."
    )

    # --- 2. Context and User Query Integration (Crucial Change Here) ---
    context_str = "\n\n".join(retrieved_context)

    # If context is empty, adjust the user message accordingly
    if not context_str.strip():
        user_content = (
            "No specific study material context was provided for this query.\n\n"
            f"Question: {user_question}\n\n"
            "Please provide your answer based *solely* on the absence of context."
        )
    else:
        # This is the key part: clearly present context, then the question, then an "Answer:" prompt.
        user_content = (
            f"Context from study materials:\n```\n{context_str}\n```\n\n"
            f"Question: {user_question}\n\n"
            "Answer:" # Explicitly cue the LLM to start answering *from the context*
        )

    # --- 3. Assemble the Full Prompt for Ollama's Chat API ---
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    return messages

def chat_with_ollama(model_name: str, messages: list[dict]):
    """
    Sends messages to the Ollama model and prints the response.

    Args:
        model_name: The name of the Ollama model to use (e.g., 'llama2:7b').
        messages: A list of message dictionaries (system, user).
    """
    try:
        print(f"\n--- Querying Ollama Model: {model_name} ---")
        print("\nSending messages:")
        for msg in messages:
            print(f"  {msg['role'].upper()}: {msg['content']}") # Print full content

        response = ollama.chat(model=model_name, messages=messages)
        print("\n--- Model Response ---")
        print(response['message']['content'])
    except ollama.ResponseError as e:
        print(f"\nError communicating with Ollama: {e}")
        print("Please ensure Ollama is running and the model is pulled (`ollama pull llama2:7b`).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# --- Configuration ---
OLLAMA_MODEL = 'llama2:7b' # Ensure this model is pulled in Ollama!

# --- Test Cases ---

# Test Case 1: Question with DIRECTLY relevant context
print("===== Test Case 1: Direct Answer Available =====")
question_1 = "What is the primary function of chloroplasts?"
context_1 = [
    "Chloroplasts are organelles found in plant cells and other eukaryotic organisms that conduct photosynthesis.",
    "They absorb sunlight and use it in conjunction with water and carbon dioxide to create food (sugar) for the plant and release oxygen."
]
messages_1 = build_llama2_rag_prompt(question_1, context_1)
chat_with_ollama(OLLAMA_MODEL, messages_1)

# Test Case 2: Question with IRRELEVANT context (simulates bad retrieval)
print("\n===== Test Case 2: Irrelevant Context =====")
question_2 = "What are the key ingredients for making sourdough bread?"
context_2 = [
    "The mitochondria is often called the 'powerhouse of the cell' because it generates most of the chemical energy needed to power the cell's biochemical reactions.",
    "ATP, adenosine triphosphate, is the main energy currency of the cell.",
    "Enzymes are biological catalysts that speed up the rate of biochemical reactions."
]
messages_2 = build_llama2_rag_prompt(question_2, context_2)
chat_with_ollama(OLLAMA_MODEL, messages_2)

# Test Case 3: Question with PARTIALLY relevant context (similar to "How does ocean help the environment?")
print("\n===== Test Case 3: Partially Relevant Context (expected 'I cannot find...') =====")
question_3 = "What are the long-term effects of volcanic eruptions on global climate?"
context_3 = [
    "Volcanic eruptions release gases and ash into the atmosphere.",
    "Large eruptions can inject sulfur dioxide into the stratosphere, which forms aerosols that reflect sunlight back into space.",
    "This reflection can lead to a temporary cooling effect on the Earth's surface."
]
messages_3 = build_llama2_rag_prompt(question_3, context_3)
chat_with_ollama(OLLAMA_MODEL, messages_3)

# Test Case 4: Question with NO context (simulates empty retrieval)
print("\n===== Test Case 4: No Context Provided =====")
question_4 = "Who was the first person to walk on the moon?"
context_4 = [] # Empty list, meaning no context was found/provided
messages_4 = build_llama2_rag_prompt(question_4, context_4)
chat_with_ollama(OLLAMA_MODEL, messages_4)