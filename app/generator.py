# app/generator.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = None
rag_chain = None


def get_llm():
    """Get or create LLM instance lazily"""
    global llm
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o")
    return llm


def get_rag_chain():
    """Get or create LCEL RAG chain lazily"""
    global rag_chain
    if rag_chain is None:
        llm = get_llm()

        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_template(
            """Answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""
        )

        # Build LCEL chain: prompt -> llm -> output parser
        rag_chain = prompt_template | llm | StrOutputParser()

    return rag_chain


def generate_answer(question: str, context: str) -> str:
    chain = get_rag_chain()

    return chain.invoke({"context": context, "question": question})
