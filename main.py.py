import nltk
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define uma lista de perguntas e respostas
questions = [
    "Secretaria Geral:",
    "Secretaria de Gestão Administrativa",
    "Compras",
    "Secretaria Infraestrutura e Logística",
    "Telefonista/Solicitação de Veículos",
    "Secretaria de Gestão de Magistrados",
    "Secretaria de Gestão de Pessoas",
    "Subsecretaria de Folha de Pagamentos",
    "Central Atendimento de TI / Informática",
    "Setor de Dados e Apoio à Decisão",
    "Subsecretaria de Infraestrutura de TIC",
    "Setor de Sistemas Administrativos",
    "Setor de Sistemas Judiciais",
    "Núcleo de Precatórios",
    "1ª Vara de Família"
    "2ª Vara de Família"
    "1ª Vara Cível"
    "2ª Vara Cível"
    "3ª Vara Cível"
    "4ª Vara Cível"
    "5ª Vara Cível"
    "1ª Vara do Tribunal do Júri e da Justiça Militar"
    "2ª Vara do Tribunal do Júri e da Justiça Militar "
    "1ª Vara Criminal"
    "2ª Vara Criminal"
    "3ª Vara Criminal"
    "Vara de Crimes contra Vulneráveis"
    "VEPEMA - Vara de Execução de Penas e Medidas Alternativas"
    "VARA DA JUSTIÇA ITINERANTE"
    ""
    
]

answers = [
    "3198 4102",
    "3198-4111",
    "3198-4145",
    "3198-4110",
    "3198-2898",
    "3198-2875",
    "3198-4151",
    "3198-4163",
    "(95) 3198-4141 - 0800 723-1783",
    "3621-5151 / 3198-4107",
    "3621-5144",
    "3621-5140",
    "3621-5140",
    "3198-4136 e 3198-4105",
    (f"Cartório: 3198-4721{os.linesep}Secretaria: 3198-4722"),
    (f"Cartório: 3198-4726{os.linesep}Secretaria: 3198-4724 "),
    (f"Cartório: 3198 4734{os.linesep}Secretaria: 3198-4754"),
    (f"Cartório: 3198-4755{os.linesep}Secretaria: 3198-4731"),
    (f"Cartório: (95) 3198-4727 / 98400-9039{os.linesep}Gabinete: (95) 3198-4728 / 98401-0490"),
    (f"Cartório: 3198- 4717{os.linesep}Secretaria: 3198-4716"),
    (f"Cartório: 3198-4719{os.linesep}Secretaria: 3198-4720"),
    (f"Cartório: 3194-2643{os.linesep}Secretaria: 3194-2650"),
    (f"Cartório: 3194-2668{os.linesep}Secretaria: 3194-2669{os.linesep}Gabinete: 3194-2670"),
    (f"Cartório: 3194-2679{os.linesep}Gabinete: 3194-2665{os.linesep}Audiência: 98404-1029"),
    (f"Cartório: 3194-2679{os.linesep}Gabinete: 3194-2608{os.linesep}Audiência: 98417-5333"),
    (f"Cartório: 3194-2679{os.linesep}Gabinete: 3194-2696{os.linesep}Audiência: 3194-2696"),
    (f"Cartório: 3194-2611{os.linesep}Gabinete: 3194-2626"),
    (f"Cartório: 3194-2656{os.linesep}Secretaria: 3194-2657"),
    (f"Fixo: 3198-4184{os.linesep}Celular/Mensagens: (95)98404-3086"),
    
    
]

# Realiza o pré-processamento dos dados
nltk.download('punkt')
nltk.download('stopwords')

tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words=nltk.corpus.stopwords.words('portuguese'))
tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

# Define uma função para obter a resposta do bot
def get_response(text):
    # Calcula a similaridade do texto com as perguntas predefinidas
    text_tfidf = tfidf_vectorizer.transform([text])
    similarities = cosine_similarity(text_tfidf, tfidf_matrix).flatten()

    # Retorna a resposta correspondente à pergunta mais similar
    index = similarities.argmax()
    return answers[index]

# Executa o loop principal do bot
while True:
    # Lê a entrada do usuário
    user_input = input("Usuário: ")

    # Obtém a resposta do bot e a imprime
    response = get_response(user_input)
    print("Chatbot: " + response)