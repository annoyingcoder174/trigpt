# src/pdf_qa.py

from pathlib import Path

from .llm_client import LLMClient
from .doc_ingestion import load_pdf_text, chunk_text
from .vector_store import add_document, query_document


def build_context_from_query(question: str, top_k: int = 5) -> str:
    """
    Use the vector store to find relevant chunks for a question
    and build a text context block.
    """
    result = query_document(question, n_results=top_k)

    # Chroma returns lists inside lists: result["documents"][0] etc.
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    context_parts: list[str] = []

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source", "unknown.pdf")
        chunk_index = meta.get("chunk_index", "?")
        context_parts.append(
            f"[Source {i} – {source}, chunk {chunk_index}]\n{doc}"
        )

    return "\n\n".join(context_parts)


def main():
    print("📄 TriGPT – PDF Q&A mode (RAG)")
    pdf_path_str = input("Enter path to a PDF file: ").strip()

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print("❌ File not found. Check the path and try again.")
        return

    print(f"🔍 Loading PDF: {pdf_path.name} ...")
    full_text = load_pdf_text(pdf_path)
    print(f"DEBUG: extracted {len(full_text)} characters from PDF.")
    print("DEBUG sample:", full_text[:500], "...\n")
    print("✂️ Splitting into chunks ...")
    chunks = chunk_text(full_text, chunk_size=800, overlap=200)
    print(f"✅ Created {len(chunks)} chunks.")

    # Use filename (without extension) as doc_id
    doc_id = pdf_path.stem
    add_document(
        doc_id,
        chunks,
        metadata={"source": pdf_path.name},
    )
    print(f"📚 Document '{pdf_path.name}' indexed in vector store.")

    client = LLMClient()

    print("\nYou can now ask questions about this PDF.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("---------------" + "\n😼PTri: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye from IreneAdler.")
            break

        context = build_context_from_query(question, top_k=5)

        system_prompt = (
            "You are an AI assistant named Irene. "
            "You are helping a user named PTri understand a PDF document.\n"
            "You must answer ONLY using the information in the provided context. "
            "If the answer is not present, say you don't know.\n\n"
            f"Here is the context from the document:\n\n{context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            answer = client.chat(messages)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            continue

        print(f"\n🤖Irene: {answer}\n")


if __name__ == "__main__":
    main()
