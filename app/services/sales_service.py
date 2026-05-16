import pandas as pd

# Importa a função experimental do LangChain responsável por criar
# um agente capaz de interagir com DataFrames do pandas.
from langchain_experimental.agents import (
    create_pandas_dataframe_agent
)

# Factory responsável por criar/configurar o modelo de linguagem (LLM)
# de acordo com o provider escolhido.
from app.core.llm_factory import LLMFactory

# Prompt customizado utilizado como instrução base do agente.
# Normalmente contém contexto, regras e comportamento esperado.
from app.prompts.templates import SALES_AGENT_PREFIX


def analyze_sales_data(
    file_path: str,
    question: str,
    provider: str = "openai"
):
    """
    Analisa dados de vendas utilizando um agente LangChain conectado
    a um DataFrame do pandas.

    A função:
    1. Carrega um arquivo CSV ou Excel.
    2. Cria um modelo de linguagem (LLM).
    3. Cria um agente LangChain especializado em DataFrames.
    4. Executa uma pergunta sobre os dados.
    5. Retorna a resposta gerada pelo agente.

    Args:
        file_path (str):
            Caminho do arquivo contendo os dados.
            Aceita arquivos .csv e .xlsx/.xls.

        question (str):
            Pergunta que será feita ao agente.
            Exemplo:
            "Qual vendedor teve maior faturamento?"

        provider (str, optional):
            Provider do modelo de linguagem.
            Exemplo: "openai", "groq", "anthropic".
            Default é "openai".

    Returns:
        str:
            Resposta gerada pelo agente com base nos dados.
    """

    # =========================
    # LOAD DATAFRAME
    # =========================
    # Nesta etapa os dados são carregados para um DataFrame pandas.
    # O pandas é a principal biblioteca Python para análise de dados.

    # Verifica se o arquivo é CSV
    if file_path.endswith(".csv"):

        # Lê o CSV e converte em DataFrame
        df = pd.read_csv(file_path)

    else:

        # Caso não seja CSV, assume que é Excel
        # read_excel suporta arquivos .xlsx e .xls
        df = pd.read_excel(file_path)

    # =========================
    # LLM
    # =========================
    # Aqui criamos o modelo de linguagem (Large Language Model).
    #
    # O LLM será responsável por:
    # - Entender a pergunta do usuário
    # - Interpretar os dados do DataFrame
    # - Decidir quais operações executar
    # - Gerar a resposta final

    llm = LLMFactory.get_llm(

        # Provider do modelo
        provider=provider,

        # Temperatura 0 = respostas mais determinísticas
        # e consistentes, ideal para análise de dados.
        temperature=0
    )

    # =========================
    # AGENT
    # =========================
    # O Agent é um dos conceitos principais do LangChain.
    #
    # Um agente:
    # - Recebe uma pergunta
    # - Decide quais tools usar
    # - Executa ações
    # - Analisa resultados intermediários
    # - Produz uma resposta final
    #
    # Neste caso usamos um agente especializado em pandas DataFrame.

    agent = create_pandas_dataframe_agent(

        # Modelo de linguagem utilizado pelo agente
        llm=llm,

        # DataFrame que o agente poderá consultar/manipular
        df=df,

        # Tipo moderno de agente do LangChain.
        #
        # "tool-calling" utiliza o mecanismo atual de tools
        # compatível com function calling dos modelos modernos.
        #
        # O agente pode:
        # - Executar código Python
        # - Fazer filtros
        # - Agrupar dados
        # - Calcular métricas
        # - Gerar análises
        agent_type="tool-calling",

        # Exibe logs detalhados da execução.
        #
        # Muito útil para debugging e aprendizado,
        # pois mostra:
        # - O raciocínio do agente
        # - Chamadas de tools
        # - Código executado
        # - Resultados intermediários
        verbose=True,

        # Permite execução de código Python.
        #
        # IMPORTANTE:
        # O agente pandas executa código dinamicamente
        # para analisar os dados.
        #
        # Deve ser usado apenas em ambientes confiáveis.
        allow_dangerous_code=True,

        # Prompt base personalizado.
        #
        # Esse prefixo geralmente contém:
        # - Contexto de negócio
        # - Regras da análise
        # - Estilo de resposta
        # - Restrições do agente
        prefix=SALES_AGENT_PREFIX,

        # Número máximo de iterações/raciocínios.
        #
        # Cada iteração pode envolver:
        # - Pensamento do agente
        # - Uso de tool
        # - Execução de código
        max_iterations=10,

        # Estratégia de parada.
        #
        # "force" interrompe o agente quando
        # atingir o limite máximo de iterações.
        early_stopping_method="force",

        # Configurações adicionais do executor.
        agent_executor_kwargs={

            # Evita falhas caso o modelo gere
            # alguma resposta mal formatada.
            #
            # Muito útil para aumentar robustez.
            "handle_parsing_errors": True
        }
    )

    # =========================
    # EXECUTION
    # =========================
    # Executa o agente enviando a pergunta do usuário.
    #
    # O método invoke():
    # - Recebe inputs
    # - Inicia o raciocínio do agente
    # - Executa tools automaticamente
    # - Retorna a resposta final

    response = agent.invoke({

        # "input" é a chave padrão esperada pelo agent executor
        "input": question
    })

    # O retorno do invoke normalmente é um dicionário.
    #
    # Exemplo:
    # {
    #     "input": "...",
    #     "output": "..."
    # }
    #
    # Aqui retornamos apenas a resposta final.
    return response["output"]