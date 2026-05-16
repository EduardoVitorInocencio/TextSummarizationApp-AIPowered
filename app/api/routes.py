import os
import shutil
import uuid

# =========================
# FASTAPI IMPORTS
# =========================
# FastAPI é um framework moderno para construção
# de APIs REST com Python.
#
# Esses imports são utilizados para:
# - criar rotas
# - receber arquivos
# - validar dados
# - tratar erros

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form,
    HTTPException
)

# =========================
# PYDANTIC
# =========================
# Pydantic é utilizado pelo FastAPI para:
# - validação de dados
# - serialização
# - tipagem automática
#
# BaseModel cria schemas fortemente tipados.

from pydantic import BaseModel

# =========================
# SERVICES
# =========================
# Importa os serviços responsáveis pela lógica de negócio.
#
# text_service:
# -> resumo de textos usando LangChain
#
# sales_service:
# -> análise de dados usando agente pandas LangChain

from app.services.text_service import summarize_text
from app.services.sales_service import analyze_sales_data


# =========================
# ROUTER
# =========================
# APIRouter permite modularizar rotas no FastAPI.
#
# Isso facilita:
# - organização
# - separação por domínio
# - escalabilidade
#
# Exemplo:
# /summarize
# /sales-insights

router = APIRouter()


# =========================
# REQUEST MODEL
# =========================
# Modelo de entrada da rota /summarize.
#
# O FastAPI usa esse model para:
# - validar JSON automaticamente
# - gerar documentação Swagger
# - garantir tipagem

class TextRequest(BaseModel):
    """
    Modelo de entrada para a rota de sumarização.

    Attributes:
        text (str):
            Texto que será resumido.

        provider (str):
            Provider do modelo de linguagem.
            Default:
                "openai"
    """

    text: str

    # Provider padrão do LLM
    provider: str = "openai"


# =========================
# ROUTE: SUMMARIZE
# =========================
# Endpoint responsável por resumir textos.
#
# Método:
# POST
#
# URL:
# /summarize
#
# Fluxo:
# 1. Recebe texto
# 2. Chama serviço LangChain
# 3. Retorna resumo

@router.post("/summarize")
async def api_summarize(request: TextRequest):
    """
    Endpoint responsável por gerar resumo de texto utilizando LangChain.

    Fluxo:
    1. Recebe texto via JSON
    2. Executa pipeline de sumarização
    3. Retorna resumo gerado pelo LLM

    Args:
        request (TextRequest):
            Dados enviados no body da requisição.

    Returns:
        dict:
            Provider utilizado e resumo gerado.

    Raises:
        HTTPException:
            Retorna erro 500 caso ocorra falha.
    """

    try:

        # Executa o serviço de sumarização.
        #
        # Internamente esse serviço:
        # - cria o LLM
        # - divide texto em chunks
        # - executa chain LangChain
        # - retorna resumo final

        result = summarize_text(

            # Texto recebido da requisição
            text=request.text,

            # Provider selecionado
            provider=request.provider
        )

        # Retorno JSON da API
        return {

            # Provider utilizado
            "provider": request.provider,

            # Resultado gerado pelo modelo
            "summary": result
        }

    except Exception as e:

        # Caso qualquer erro aconteça,
        # retornamos HTTP 500.

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


# =========================
# ROUTE: SALES INSIGHTS
# =========================
# Endpoint responsável por análise de planilhas.
#
# Método:
# POST
#
# URL:
# /sales-insights
#
# O endpoint recebe:
# - arquivo CSV/Excel
# - pergunta
# - provider
#
# Depois:
# - salva arquivo temporariamente
# - cria agente LangChain pandas
# - executa análise
# - retorna insight

@router.post("/sales-insights")
async def api_sales_insights(

    # Pergunta enviada via multipart/form-data
    question: str = Form(...),

    # Provider opcional
    provider: str = Form("openai"),

    # Arquivo enviado na requisição
    file: UploadFile = File(...)

):
    """
    Endpoint responsável por gerar insights de vendas
    utilizando LangChain + Pandas Agent.

    Fluxo:
    1. Recebe arquivo CSV/Excel
    2. Salva arquivo temporariamente
    3. Cria agente pandas
    4. Executa pergunta sobre os dados
    5. Retorna insight gerado pelo LLM

    Args:
        question (str):
            Pergunta sobre os dados.

        provider (str):
            Provider do modelo de linguagem.

        file (UploadFile):
            Arquivo enviado pelo usuário.

    Returns:
        dict:
            Resultado da análise.

    Raises:
        HTTPException:
            Retorna erro 500 caso ocorra falha.
    """

    # Variável usada para controle do arquivo temporário
    temp_file_path = None

    try:

        # =========================
        # FILE EXTENSION
        # =========================
        # Extrai extensão do arquivo.
        #
        # Exemplo:
        # vendas.csv -> csv

        extension = file.filename.split(".")[-1]

        # =========================
        # TEMP FILE
        # =========================
        # Cria nome único utilizando UUID.
        #
        # uuid4():
        # gera identificador aleatório.
        #
        # Isso evita conflito entre uploads.

        temp_file_path = (
            f"temp_{uuid.uuid4()}.{extension}"
        )

        # =========================
        # SAVE FILE
        # =========================
        # Salva o arquivo enviado temporariamente.
        #
        # "wb":
        # write binary

        with open(temp_file_path, "wb") as buffer:

            # copyfileobj copia conteúdo do upload
            # para o arquivo temporário

            shutil.copyfileobj(

                file.file,

                buffer
            )

        # =========================
        # SALES ANALYSIS
        # =========================
        # Executa o agente LangChain pandas.
        #
        # Internamente:
        # - carrega DataFrame
        # - cria LLM
        # - cria Agent
        # - executa tools
        # - responde pergunta

        result = analyze_sales_data(

            # Caminho do arquivo salvo
            file_path=temp_file_path,

            # Pergunta enviada pelo usuário
            question=question,

            # Provider selecionado
            provider=provider
        )

        # =========================
        # API RESPONSE
        # =========================
        # Retorno JSON da API

        return {

            # Provider utilizado
            "provider": provider,

            # Pergunta realizada
            "question": question,

            # Insight gerado pelo agente
            "insight": result
        }

    except Exception as e:

        # Tratamento de erro genérico

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )

    finally:

        # =========================
        # CLEANUP
        # =========================
        # Remove o arquivo temporário ao final.
        #
        # finally sempre executa:
        # - com erro
        # - sem erro
        #
        # Isso evita acúmulo de arquivos.

        if (

            temp_file_path

            and os.path.exists(temp_file_path)
        ):

            os.remove(temp_file_path)