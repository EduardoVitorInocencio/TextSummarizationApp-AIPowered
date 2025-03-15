# README - Resumo de Texto com LangChain e Anthropic

Este repositório contém um script em Python que utiliza o **LangChain** e a API **Anthropic** para realizar a sumarização de um texto. Abaixo, explicamos em detalhes cada parte do código, como configurar o ambiente, instalar dependências e executar o script.

## Índice

1. [Introdução](#introdução)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação e Execução](#instalação-e-execução)
4. [Estrutura do Código](#estrutura-do-código)
    1. [Importação de Bibliotecas](#importação-de-bibliotecas)
    2. [Carregamento de Variáveis de Ambiente](#carregamento-de-variáveis-de-ambiente)
    3. [Criação do Modelo AI](#criação-do-modelo-ai)
    4. [Fatiamento de Texto](#fatiamento-de-texto)
    5. [Criação de Documentos](#criação-de-documentos)
    6. [Summarização dos Textos](#summarização-dos-textos)
5. [Conclusão](#conclusão)

## Introdução

Este script utiliza a biblioteca **LangChain** e a API **Anthropic** para dividir um texto em partes menores e realizar a sumarização. O texto é fornecido diretamente no código, e o modelo de IA **Claude-3** da Anthropic é utilizado para gerar o resumo.

## Pré-requisitos

Antes de rodar o código, é necessário garantir que você tem os seguintes pré-requisitos instalados:

- Python 3.7 ou superior.
- Pacotes de Python: `langchain`, `dotenv`, `anthropic`, entre outros.
- Uma chave de API da **Anthropic**.

## Instalação e Execução

Siga as etapas abaixo para configurar o ambiente e executar o script:

1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   ```

2. Navegue até a pasta do projeto:
   ```bash
   cd <PASTA_DO_REPOSITORIO>
   ```

3. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```

4. Ative o ambiente virtual:
   - **No Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **No macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

5. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```

6. Crie um arquivo `.env` na raiz do projeto e adicione a sua chave de API da **Anthropic**:
   ```bash
   ANTHROPIC_API_KEY=your_api_key_here
   ```

7. Execute o script:
   ```bash
   python script.py
   ```

## Estrutura do Código

### 1. Importação de Bibliotecas

```python
from langchain_anthropic import ChatAnthropic                   # Importando a biblioteca anthropic
from langchain.docstore.document import Document                # Importando a biblioteca Document permite a criação de documentos
from langchain.text_splitter import CharacterTextSplitter       # Importando a biblioteca CharacterTextSplitter que permite a divisão de texto
from langchain.chains.summarize import load_summarize_chain     # Importando a biblioteca load_summarize_chain que permite a sumarização de texto
from dotenv import load_dotenv, find_dotenv                     # Importando a biblioteca dotenv que permite a leitura de variáveis de ambiente
import os                                                       # Importando a biblioteca os que permite a manipulação de variáveis de ambiente
```

Essas bibliotecas são responsáveis por realizar as principais funcionalidades do script, como a integração com a API da Anthropic, divisão do texto e manipulação de variáveis de ambiente.

### 2. Carregamento de Variáveis de Ambiente

```python
load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
```

Aqui, carregamos as variáveis de ambiente a partir do arquivo `.env`. A chave de API da Anthropic (`ANTHROPIC_API_KEY`) é obtida e armazenada em uma variável para ser usada na configuração do modelo de IA.

### 3. Criação do Modelo AI

```python
llm = ChatAnthropic(
    model = "claude-3-opus-20240229",
    temperature= 0, # Ajusta o nível de criatividade do modelo
    anthropic_api_key = ANTHROPIC_API_KEY
)
```

Neste bloco, é criada uma instância do modelo de linguagem da Anthropic. O modelo `claude-3-opus-20240229` é configurado com uma temperatura de `0`, o que significa que o modelo será mais preciso e menos criativo.

### 4. Fatiamento de Texto

```python
text_splitter = CharacterTextSplitter()
texts = text_splitter.split_text(text)
```

Aqui, o texto fornecido é dividido em partes menores utilizando a classe `CharacterTextSplitter`. Isso permite que o modelo trabalhe com textos grandes, quebrando-os em fragmentos que podem ser processados.

### 5. Criação de Documentos

```python
docs = [Document(page_content=text) for text in texts]  # List Comprenhension
```

Com o texto dividido, criamos uma lista de objetos `Document`, que contêm o conteúdo de cada parte do texto. Isso facilita a manipulação e o processamento do texto no LangChain.

### 6. Summarização dos Textos

```python
chain = load_summarize_chain(llm=llm, chain_type="stuff")
summary = chain.invoke(docs)  # Executa a cadeia de resumo dos textos
```

A função `load_summarize_chain` é usada para carregar a cadeia de sumarização, configurando o modelo de linguagem e o tipo de cadeia (`"stuff"`). O método `invoke` é então chamado para gerar o resumo a partir dos documentos processados.

### Exibição do Resumo

```python
print(summary['output_text'])  # Exibe o resumo do texto
```

Por fim, o resumo gerado é impresso na tela.

## Conclusão

Esse script demonstra como integrar a API da **Anthropic** com o **LangChain** para realizar a sumarização de textos longos. Ao seguir as instruções acima, você pode configurar e executar o código para testar a funcionalidade de resumo automatizado.

Se você tiver dúvidas ou sugestões, fique à vontade para abrir uma **issue** no repositório.