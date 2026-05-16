from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm_factory import LLMFactory
from prompts.templates import summary_prompt

def summarize_text(text: str,provider: str = "openai") -> str:

    llm = LLMFactory.get_llm(provider)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    chain = (
        summary_prompt
        | llm
        | StrOutputParser()
    )

    partial_summaries = []

    for chunk in chunks:

        response = chain.invoke({
            "text": chunk
        })

        partial_summaries.append(response)

    final_text = "\n".join(partial_summaries)

    final_summary = chain.invoke({
        "text": final_text
    })

    return final_summary