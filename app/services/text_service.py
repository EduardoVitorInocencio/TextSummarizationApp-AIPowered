from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.llm_factory import LLMFactory
from app.prompts.templates import summary_prompt


def summarize_text(text: str, provider: str = "openai") -> str:
    """
    Gera um resumo de um texto utilizando LangChain e um modelo de linguagem (LLM).

    O fluxo da função funciona em etapas:

    1. Cria o modelo de linguagem (LLM)
    2. Divide textos grandes em pequenos chunks
    3. Cria uma chain LangChain
    4. Resume cada chunk individualmente
    5. Junta os resumos parciais
    6. Gera um resumo final consolidado

    Essa abordagem é muito utilizada para evitar problemas
    de limite de tokens dos modelos.

    Args:
        text (str):
            Texto completo que será resumido.

        provider (str, optional):
            Provider do modelo de linguagem.
            Exemplos:
            - "openai"
            - "groq"
            - "anthropic"

            Default:
                "openai"

    Returns:
        str:
            Resumo final gerado pelo modelo.
    """

    # =========================
    # LLM
    # =========================
    # Cria o modelo de linguagem usando a factory da aplicação.
    #
    # O LLM (Large Language Model) será responsável
    # por interpretar e resumir o texto.

    llm = LLMFactory.get_llm(provider)

    # =========================
    # TEXT SPLITTER
    # =========================
    # O RecursiveCharacterTextSplitter é um splitter
    # oficial do LangChain utilizado para quebrar textos grandes.
    #
    # Isso é necessário porque modelos possuem limite
    # de tokens/contexto.
    #
    # O splitter tenta dividir o texto preservando
    # contexto e estrutura semântica.

    splitter = RecursiveCharacterTextSplitter(

        # Quantidade máxima de caracteres por chunk
        chunk_size=2000,

        # Quantidade de sobreposição entre chunks.
        #
        # Isso ajuda a manter contexto entre partes
        # consecutivas do texto.
        chunk_overlap=200
    )

    # Divide o texto em vários pedaços menores (chunks)
    chunks = splitter.split_text(text)

    # =========================
    # CHAIN
    # =========================
    # Chain é um dos conceitos principais do LangChain.
    #
    # Uma chain conecta componentes em sequência.
    #
    # Neste caso:
    #
    # Prompt -> LLM -> Parser
    #
    # Fluxo:
    # 1. O prompt recebe os dados
    # 2. O LLM gera a resposta
    # 3. O parser converte a saída

    chain = (

        # Prompt template com instruções de resumo
        summary_prompt

        # Operador "|" cria pipelines no LangChain Expression Language (LCEL)
        #
        # O resultado do prompt é enviado para o LLM
        | llm

        # Converte a saída do modelo em string simples
        | StrOutputParser()
    )

    # Lista que armazenará os resumos parciais
    partial_summaries = []

    # =========================
    # PARTIAL SUMMARIZATION
    # =========================
    # Aqui resumimos cada chunk individualmente.
    #
    # Essa estratégia é chamada de:
    #
    # "Map Reduce Summarization"
    #
    # Primeiro:
    # - resume partes menores
    #
    # Depois:
    # - combina os resultados

    for chunk in chunks:

        # Executa a chain enviando o chunk atual
        #
        # invoke() é o método padrão de execução
        # de chains no LangChain.
        response = chain.invoke({

            # Variável esperada pelo prompt template
            "text": chunk
        })

        # Adiciona o resumo parcial na lista
        partial_summaries.append(response)

    # Junta todos os resumos parciais em um único texto
    final_text = "\n".join(partial_summaries)

    # =========================
    # FINAL SUMMARIZATION
    # =========================
    # Agora resumimos novamente os resumos parciais.
    #
    # Isso gera um resumo final mais compacto
    # e consolidado.

    final_summary = chain.invoke({

        # Envia o texto consolidado para resumo final
        "text": final_text
    })

    # Retorna o resumo final
    return final_summary