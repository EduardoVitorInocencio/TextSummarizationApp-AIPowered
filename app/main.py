# IMPORTANDO AS BIBLIOTECAS
from langchain_anthropic import ChatAnthropic                   # Importando a biblioteca anthropic
from langchain.docstore.document import Document                # Importando a biblioteca Document permite a criação de documentos
from langchain.text_splitter import CharacterTextSplitter       # Importando a biblioteca CharacterTextSplitter que permite a divisão de texto
from langchain.chains.summarize import load_summarize_chain     # Importando a biblioteca load_summarize_chain que permite a sumarização de texto
from dotenv import load_dotenv, find_dotenv                     # Importando a biblioteca dotenv que permite a leitura de variáveis de ambiente
import os                                                       # Importando a biblioteca os que permite a manipulação de variáveis de ambiente

# CARREGANDO AS VARIÁVEIS DE AMBIENTE
load_dotenv(find_dotenv())
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# CRIAR MODELO AI
llm = ChatAnthropic(
    model = "claude-3-opus-20240229",
    temperature= 0, # Ajusta o nível de criatividade do modelo
    anthropic_api_key = ANTHROPIC_API_KEY
)

text = """
São Jorge (em grego: Άγιος Γεώργιος; romaniz.: Ágios Geṓrgios; em latim: Georgius; entre 275 e 280 — 23 de abril de 303), também conhecido como Jorge da Capadócia e Jorge de Lida foi, conforme a tradição, um soldado romano no exército do imperador Diocleciano, venerado como mártir cristão. Na hagiografia, São Jorge é um dos santos mais venerados no Catolicismo, na Igreja Ortodoxa, bem como na Comunhão Anglicana. É imortalizado na lenda em que mata o dragão. É também um dos catorze santos auxiliares. No cânon do Papa Gelásio (496 d.C.), São Jorge é mencionado entre aqueles que “foram justamente reverenciados pelos homens e cujos atos são conhecidos somente por Deus”.[3]

Considerado um dos mais proeminentes santos militares, a memória de São Jorge é celebrada nos dias 23 de abril e 3 de novembro – a primeira é a data de sua morte, e a segunda, a data da consagração da igreja dedicada a ele em Lida (Israel), na qual se encontram sua sepultura e suas relíquias. A igreja de São Jorge em Lida foi erguida a mando do imperador romano Constantino.

São Jorge é o santo padroeiro em diversas partes do mundo, tais como Inglaterra, Geórgia, Lituânia, Sérvia, Montenegro e Etiópia, além de ser um padroeiro menor de Portugal, bem como das seguintes cidades: Londres, Barcelona, Gênova, Régio da Calábria, Ferrara, Friburgo, Moscou e Beirute. No Oriente, ele é conhecido como “megalomártir”, ou seja, “grande mártir”. Ele também é reconhecido como modelo de homem virgem, ao lado de São João Evangelista e do próprio Jesus Cristo.[3]

Há uma tradição que aponta o ano 303 como ano da sua morte. Apesar de sua história se basear em narrativas lendários e apócrifas (decreto gelasiano do século VI), a devoção a São Jorge se espalhou por todo o mundo.
"""

# SPLIT TEXT : Fatiamento do Texto
text_splitter = CharacterTextSplitter()
texts = text_splitter.split_text(text)

# CREATE DOCUMENTS : CRIAÇÃO DE DOCUMENTOS
docs = [Document(page_content=text) for text in texts] # List Comprenhension

# Summarização dos textos (Prompt & Chain (Função da LLM))
chain = load_summarize_chain(llm=llm, chain_type="stuff")

#Executar a Chain
summary = chain.invoke(docs) # Executa a cadeia de resumo dos textos]

print(summary['output_text']) # Exibe o resumo do texto

