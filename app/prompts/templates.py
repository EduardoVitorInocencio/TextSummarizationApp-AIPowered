from langchain_core.prompts import PromptTemplate

# Prompt para Sumarização
SUMMARY_TEMPLATE = """
    Escreva um resumo conciso e direto do seguinte texto:
    "{text}"
    RESUMO CONCISO:
"""
summary_prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["text"])

# Prompt Base para o Agente de Vendas (Opcional, pois o agente Pandas tem o seu próprio, mas podemos customizar)
SALES_AGENT_PREFIX = """
    Você é um analista de dados especialista em vendas.
    Trabalhe com o dataframe pandas fornecido.
    Forneça insights estratégicos, tendências e respostas diretas para as perguntas de negócios.
    Sempre responda em Português do Brasil.
"""