# README - Projeto de Estudos com LangChain + Multi Providers (OpenAI, Groq e mais)

## Introdução

Este projeto é um ambiente de estudos pessoais focado em **LLMs (Large Language Models)** utilizando o ecossistema do [LangChain](https://www.langchain.com/?utm_source=chatgpt.com).

O objetivo principal é aprender na prática como integrar múltiplos providers de IA generativa em uma única aplicação Python, explorando:

* Integração com diferentes modelos de linguagem
* Criação de agentes com LangChain
* Análise de arquivos e planilhas
* Sumarização de textos
* Engenharia de prompts
* APIs com FastAPI
* Arquitetura escalável para múltiplos providers

Atualmente o projeto já possui integração com:

* [OpenAI](https://platform.openai.com/?utm_source=chatgpt.com)
* [Groq](https://groq.com/?utm_source=chatgpt.com)

E está preparado para futura expansão com:

* [Google Gemini](https://ai.google.dev/?utm_source=chatgpt.com)
* [DeepSeek](https://www.deepseek.com/?utm_source=chatgpt.com)
* [Llama (Meta)](https://www.llama.com/?utm_source=chatgpt.com)

---

# Tecnologias Utilizadas

## Backend

* [Python](https://www.python.org/?utm_source=chatgpt.com)
* [FastAPI](https://fastapi.tiangolo.com/?utm_source=chatgpt.com)
* [LangChain](https://python.langchain.com/?utm_source=chatgpt.com)

## Providers de IA

* [OpenAI API](https://platform.openai.com/docs/overview?utm_source=chatgpt.com)
* [Groq API](https://console.groq.com/docs/overview?utm_source=chatgpt.com)

## Futuras Integrações

* [Google Gemini API](https://ai.google.dev/docs?utm_source=chatgpt.com)
* [DeepSeek API](https://api-docs.deepseek.com/?utm_source=chatgpt.com)
* [Meta Llama](https://ai.meta.com/llama/?utm_source=chatgpt.com)

---

# Objetivos do Projeto

Este projeto foi criado para:

* Estudar arquitetura multi-provider para LLMs
* Aprender LangChain na prática
* Trabalhar com Agents
* Explorar análise de dados com IA
* Construir APIs modernas com FastAPI
* Testar modelos diferentes utilizando a mesma interface
* Comparar performance entre providers
* Criar uma base reutilizável para futuros projetos de IA

---

# Funcionalidades Atuais

## Sumarização de Texto

Permite resumir textos longos utilizando modelos de IA.

### Recursos:

* Divisão automática de textos grandes
* Criação de documentos com LangChain
* Cadeias de sumarização
* Suporte a múltiplos modelos

---

## Análise de Planilhas com IA

Utilizando:

* Pandas
* LangChain Agents
* DataFrame Agents

O sistema consegue:

* Ler planilhas CSV/XLSX
* Responder perguntas sobre os dados
* Gerar análises automáticas
* Criar insights com IA

---

## Arquitetura Multi Provider

O projeto utiliza uma fábrica de LLMs (`LLMFactory`) para abstrair os providers.

Exemplo:

```python
llm = LLMFactory.create(provider="openai")
```

Ou:

```python
llm = LLMFactory.create(provider="groq")
```

Isso facilita:

* troca de modelos
* manutenção
* escalabilidade
* testes entre providers

---

# Estrutura do Projeto

```bash
project/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   └── llm_factory.py
│   │
│   ├── prompts/
│   │   └── templates.py
│   │
│   ├── services/
│   │   ├── summarize_service.py
│   │   └── dataframe_service.py
│   │
│   └── main.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# Como Funciona o Projeto

## 1. Carregamento das Variáveis de Ambiente

O projeto utiliza `python-dotenv` para carregar as chaves da API.

Exemplo:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 2. Factory Pattern para LLMs

A aplicação utiliza uma fábrica responsável por criar o provider correto.

Exemplo simplificado:

```python
class LLMFactory:

    @staticmethod
    def create(provider: str):

        if provider == "openai":
            ...

        elif provider == "groq":
            ...
```

Benefícios:

* desacoplamento
* escalabilidade
* fácil manutenção
* suporte futuro para novos providers

---

## 3. Text Splitter

Para textos grandes:

```python
from langchain.text_splitter import CharacterTextSplitter
```

O texto é dividido em partes menores para processamento eficiente.

---

## 4. Documentos LangChain

```python
from langchain.docstore.document import Document
```

Os textos são transformados em documentos manipuláveis pelo LangChain.

---

## 5. Chains de Sumarização

```python
from langchain.chains.summarize import load_summarize_chain
```

A chain executa a sumarização utilizando o modelo configurado.

---

## 6. Agents com Pandas

Utilizando:

```python
create_pandas_dataframe_agent
```

O agente consegue interagir diretamente com DataFrames usando linguagem natural.

---

# Pré-requisitos

Antes de executar o projeto, você precisa ter instalado:

* Python 3.10+
* Pip
* Virtualenv (opcional, mas recomendado)

---

# Instalação

## 1. Clone o Repositório

```bash
git clone <URL_DO_REPOSITORIO>
```

---

## 2. Entre na Pasta

```bash
cd <NOME_DO_PROJETO>
```

---

## 3. Crie um Ambiente Virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Instale as Dependências

```bash
pip install -r requirements.txt
```

---

# Configuração do `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

Futuras integrações:

```env
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
LLAMA_API_KEY=your_llama_key
```

---

# Executando a API

## Desenvolvimento

```bash
uvicorn app.main:app --reload
```

---

# Documentação Automática da API

Após iniciar a aplicação:

## Swagger UI

[Swagger Docs](http://localhost:8000/docs?utm_source=chatgpt.com)

## ReDoc

[ReDoc Docs](http://localhost:8000/redoc?utm_source=chatgpt.com)

---

# Exemplo de Health Check

## Endpoint

```http
GET /
```

## Resposta

```json
{
  "status": "online"
}
```

---

# Exemplo de Sumarização

```python
summary = chain.invoke(docs)
```

---

# Exemplo de Análise de Dados

```python
response = agent.invoke(question)
```

---

# Providers Suportados

| Provider | Status |
| -------- | ------ |
| OpenAI   | ✅      |
| Groq     | ✅      |
| Gemini   | 🚧     |
| DeepSeek | 🚧     |
| Llama    | 🚧     |

---

# Conceitos Estudados no Projeto

* LangChain
* Prompt Engineering
* Chains
* Agents
* Data Analysis Agents
* FastAPI
* Factory Pattern
* Multi Provider Architecture
* Environment Variables
* Text Splitters
* AI APIs
* LLM Abstraction

---

# Melhorias Futuras

* [ ] Integração com Gemini
* [ ] Integração com DeepSeek
* [ ] Integração com Llama
* [ ] Upload de arquivos via API
* [ ] Banco de dados
* [ ] Memória conversacional
* [ ] RAG (Retrieval Augmented Generation)
* [ ] Vector Database
* [ ] Streaming de respostas
* [ ] Logs e observabilidade
* [ ] Docker
* [ ] Testes automatizados
* [ ] Deploy em cloud

---

# Aprendizados

Esse projeto serve como laboratório pessoal para estudar:

* Arquitetura de aplicações com IA
* Integração entre múltiplos modelos
* Engenharia de prompts
* Criação de APIs modernas
* Escalabilidade para aplicações LLM-based

---

# Referências

* [LangChain Documentation](https://python.langchain.com/docs/introduction/?utm_source=chatgpt.com)
* [FastAPI Documentation](https://fastapi.tiangolo.com/?utm_source=chatgpt.com)
* [OpenAI Platform](https://platform.openai.com/docs?utm_source=chatgpt.com)
* [Groq Documentation](https://console.groq.com/docs?utm_source=chatgpt.com)

---

# Licença

Este projeto é apenas para fins educacionais e estudos pessoais.
