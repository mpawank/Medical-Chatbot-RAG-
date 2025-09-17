# fixed_rag_groq.py
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()  # make sure .env has GROQ_API_KEY=...
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are **MediBot**, an AI assistant trained to help users understand medical documents and health-related questions.

Use the provided CONTEXT to answer the user's QUESTION. If you don't know, say you don't know. Be concise and clear.

---
Context:
{context}

User Question:
{question}

Answer:
- Respond calmly, factually, respectfully.
- Use simple language and step-by-step only when needed.
"""
)


def get_llm_chain(retriever):
    """
    Returns a RetrievalQA chain wired to Groq via langchain-groq.
    retriever: a LangChain retriever (e.g., vectorstore.as_retriever()).
    """

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    # Create ChatGroq LLM. Use model_name (no leading whitespace).
    # Some wrappers accept "groq/..." prefixes; "llama3-70b-8192" is the Groq model id.
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",  # or "groq/llama3-70b-8192" depending on wrapper usage
        temperature=0.0,
        streaming=False,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff", "map_rerank", "refine" are common choices
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=False,
    )

    return qa


# Example usage (assuming `my_retriever` is ready):
# qa_chain = get_llm_chain(my_retriever)
# # Option A: run with .run(...) to get just the answer string
# answer = qa_chain.run("What are common causes of chest pain?")
# print(answer)
#
# # Option B: call the chain to get full outputs including source docs
# out = qa_chain({"query": "What are common causes of chest pain?"})
# print(out["result"])
# sources = out.get("source_documents", [])
# for d in sources:
#     print("SOURCE:", d.metadata.get("source"), " --- ", d.page_content[:200])
