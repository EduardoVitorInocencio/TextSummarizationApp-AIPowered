from langchain_core.prompts import PromptTemplate

# =========================
# PROMPT PARA SUMARIZAÇÃO
# =========================
# Aqui estamos criando um template de prompt.
#
# Prompt é uma instrução enviada para o modelo de linguagem (LLM).
#
# O objetivo deste prompt é orientar o modelo
# a gerar um resumo curto e objetivo.
#
# O texto "{text}" será substituído dinamicamente
# pelo conteúdo enviado durante a execução.

SUMMARY_TEMPLATE = """
    Escreva um resumo conciso e direto do seguinte texto:
    "{text}"
    RESUMO CONCISO:
"""

# =========================
# PROMPT TEMPLATE
# =========================
# PromptTemplate é uma classe do LangChain utilizada
# para criar prompts dinâmicos.
#
# Ela permite:
# - reutilizar prompts
# - inserir variáveis
# - padronizar entradas
#
# input_variables define quais variáveis
# o template espera receber.
#
# Neste caso:
# - "text" será substituído no {text}

summary_prompt = PromptTemplate(

    # Template do prompt
    template=SUMMARY_TEMPLATE,

    # Variáveis esperadas no template
    input_variables=["text"]
)

# =========================
# PROMPT BASE DO AGENTE
# =========================
# Esse prompt funciona como instrução principal
# para o agente LangChain.
#
# Ele define:
# - comportamento
# - especialização
# - idioma
# - contexto
#
# Esse tipo de prompt é muito utilizado para:
# - personalizar agentes
# - controlar respostas
# - limitar comportamento
# - adicionar contexto de negócio
#
# No caso do agente pandas:
# o LangChain já possui prompts internos,
# porém aqui estamos adicionando um prefixo customizado.

SALES_AGENT_PREFIX = """
    Você é um analista de dados especialista em vendas.

    Trabalhe com o dataframe pandas fornecido.

    Forneça insights estratégicos, tendências e respostas diretas para as perguntas de negócios.

    Sempre responda em Português do Brasil.
"""

# =========================
# COMO ESSE PROMPT É USADO
# =========================
# No create_pandas_dataframe_agent:
#
# agent = create_pandas_dataframe_agent(
#     ...
#     prefix=SALES_AGENT_PREFIX
# )
#
# O LangChain combina:
# - prompt interno do agente
# - tools disponíveis
# - contexto do dataframe
# - prefixo customizado
#
# Isso influencia diretamente:
# - estilo da resposta
# - idioma
# - tipo de análise
# - comportamento do agente