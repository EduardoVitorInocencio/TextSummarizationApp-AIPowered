import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

class LLMFactory:
    @staticmethod
    def get_llm(provider: str, temperature: float = 0.7, max_tokens: int = 2048):
        if provider == "openai":
            return ChatOpenAI(
                model="gpt-5.4-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "groq":
            return ChatGroq(
                model="groq/compound-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")