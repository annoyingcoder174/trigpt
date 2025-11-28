# src/main.py

from .llm_client import LLMClient


def main():
    print("🔹 TriGPT Local Test – type 'exit' to quit")
    client = LLMClient()

    system_prompt = (
        "You are a helpful AI assistant. "
        "Be concise and clear in your answers."
    )

    # We keep the conversation history so the model has context
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        try:
            user_input = input("---------------" + "\n😼PTri: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            reply = client.chat(messages)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # optional: remove the last user message if call fails
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"🤖Irene: {reply}")


if __name__ == "__main__":
    main()
