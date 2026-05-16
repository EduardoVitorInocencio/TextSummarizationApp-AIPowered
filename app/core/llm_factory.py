import os

# Importa funções do python-dotenv para carregar variáveis
# de ambiente a partir do arquivo .env
from dotenv import load_dotenv, find_dotenv

# Integração LangChain com modelos da OpenAI
from langchain_openai import ChatOpenAI

# Integração LangChain com modelos da Groq
from langchain_groq import ChatGroq


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================
# find_dotenv():
# Procura automaticamente o arquivo .env no projeto.
#
# load_dotenv():
# Carrega as variáveis do .env para o ambiente da aplicação.
#
# Isso permite acessar:
# - OPENAI_API_KEY
# - GROQ_API_KEY
# - outras configurações sensíveis
#
# Exemplo de .env:
#
# OPENAI_API_KEY=xxxxx
# GROQ_API_KEY=xxxxx

load_dotenv(find_dotenv())


class LLMFactory:
    """
    Factory responsável por criar instâncias de modelos de linguagem (LLMs).

    O objetivo da Factory é centralizar:
    - configuração dos modelos
    - providers suportados
    - parâmetros padrão
    - gerenciamento de API Keys

    Isso facilita:
    - manutenção
    - troca de provider
    - reutilização de código
    - escalabilidade

    Providers atualmente suportados:
    - OpenAI
    - Groq
    """

    @staticmethod
    def get_llm(
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Cria e retorna uma instância de modelo de linguagem (LLM)
        de acordo com o provider informado.

        Essa função encapsula a criação dos modelos LangChain,
        permitindo trocar providers facilmente sem alterar
        o restante da aplicação.

        Args:
            provider (str):
                Nome do provider.

                Valores suportados:
                - "openai"
                - "groq"

            temperature (float, optional):
                Controla criatividade do modelo.

                Valores baixos:
                - respostas mais determinísticas
                - mais previsíveis

                Valores altos:
                - respostas mais criativas
                - maior variação

                Default:
                    0.7

            max_tokens (int, optional):
                Quantidade máxima de tokens da resposta.

                Default:
                    2048

        Returns:
            BaseChatModel:
                Instância configurada do modelo LangChain.

        Raises:
            ValueError:
                Caso o provider informado não seja suportado.
        """

        # =========================
        # OPENAI
        # =========================
        # Cria um modelo ChatOpenAI do LangChain.
        #
        # ChatOpenAI é o wrapper oficial LangChain
        # para modelos da OpenAI.
        #
        # O modelo utilizado:
        # - gpt-4o-mini
        #
        # Esse objeto será usado posteriormente
        # em:
        # - chains
        # - agents
        # - tools
        # - pipelines LCEL

        if provider == "openai":

            return ChatOpenAI(

                # Modelo da OpenAI
                model="gpt-4o-mini",

                # Criatividade/controlabilidade
                temperature=temperature,

                # Limite máximo de tokens da resposta
                max_tokens=max_tokens,

                # API Key carregada do .env
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # =========================
        # GROQ
        # =========================
        # ChatGroq é a integração LangChain
        # para modelos servidos pela Groq.
        #
        # A Groq oferece inferência extremamente rápida
        # para modelos LLM.
        #
        # O funcionamento no LangChain é semelhante
        # ao ChatOpenAI.

        elif provider == "groq":

            return ChatGroq(

                # Modelo hospedado pela Groq
                model="groq/compound-mini",

                # Criatividade das respostas
                temperature=temperature,

                # Limite máximo de tokens
                max_tokens=max_tokens,

                # API Key do provider
                api_key=os.getenv("GROQ_API_KEY")
            )

        # =========================
        # PROVIDER NÃO SUPORTADO
        # =========================
        # Caso seja informado um provider inválido,
        # lançamos um erro explícito.
        #
        # Isso ajuda:
        # - debugging
        # - validação
        # - manutenção

        else:

            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )