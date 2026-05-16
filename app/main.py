from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="LLM Multi Provider API",
    version="2.0.0",
    description="""
    API para:
    - sumarização
    - análise de planilhas
    - multi providers (OpenAI/Groq)
"""
)

app.include_router(
    router,
    prefix="/api/v1"
)


@app.get("/")
def health_check():

    return {
        "status": "online"
    }