import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import spacy
import os

# Baixar recursos do NLTK se ainda não tiver feito
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Carregar o modelo de português do spaCy
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    print("Baixando o modelo de português do spaCy...")
    from spacy.cli import download
    download("pt_core_news_sm")
    nlp = spacy.load('pt_core_news_sm')

def preprocess_text(text, is_doril=False):
    """
    Função para pré-processar um texto aplicando as transformações solicitadas.
    """
    results = {}
    current_text = text

    # 1. Remover tags HTML
    current_text = re.sub(r'<.*?>', '', current_text)
    results['1. Remover tags HTML'] = current_text

    # 2. Remover URLs
    current_text = re.sub(r'http\S+|www\S+|t.co/\S+', '', current_text)
    results['2. Remover URLs'] = current_text

    # 3. Remover emojis
    current_text = emoji.demojize(current_text)
    current_text = re.sub(r':\S+?:', '', current_text)
    current_text = current_text.replace('😹😹', '').replace('😴😴', '').replace('🙌🙌🙌', '')
    results['3. Remover emojis'] = current_text

    # 4. Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    words = current_text.split()
    current_text = ' '.join([word for word in words if word.lower() not in stop_words])
    results['4. Remover stopwords'] = current_text

    # 5. Remover sinais de pontuação
    current_text = re.sub(r'[.,?!:;\'"()\[\]<>{}|]', '', current_text)
    results['5. Remover sinais de pontuação'] = current_text

    # 6. Remover caracteres especiais
    current_text = re.sub(r'[˃α≤]', '', current_text)
    current_text = re.sub(r'[^a-zA-Z0-9\s]', '', current_text)
    results['6. Remover caracteres especiais'] = current_text

    # 7. Remover espaços em branco excedentes
    current_text = re.sub(r'\s+', ' ', current_text).strip()
    results['7. Remover espaços em branco excedentes'] = current_text

    # 8. Substituir palavras usadas em chat por suas formas normais
    chat_dict = {
        'vc': 'você', 'qto': 'quanto', 'eh': 'é', 'tb': 'também', 'pra': 'para',
        'so': 'só', 'kk': 'risada'
    }
    words = current_text.split()
    current_text = ' '.join([chat_dict.get(word.lower(), word) for word in words])
    results['8. Substituir palavras usadas em chat'] = current_text

    # 9. Converter números em palavras
    number_words_dict = {
        '1': 'um', '2': 'dois', '3': 'três', '4': 'quatro', '5': 'cinco', '6': 'seis', '7': 'sete',
        '8': 'oito', '9': 'nove', '10': 'dez', '11': 'onze', '12': 'doze', '13': 'treze', '14': 'catorze',
        '15': 'quinze', '18': 'dezoito', '20': 'vinte', '22': 'vinte e dois', '23': 'vinte e três',
        '25': 'vinte e cinco', '28': 'vinte e oito', '29': 'vinte e nove', '30': 'trinta', '32': 'trinta e dois',
        '39': 'trinta e nove', '40': 'quarenta', '47': 'quarenta e sete', '72': 'setenta e dois',
        '500': 'quinhentos', '1500': 'mil e quinhentos', '33236333': 'trinta e três vinte e três sessenta e três trinta e três',
        '26236': 'vinte e seis duzentos e trinta e seis'
    }
    words = current_text.split()
    current_text = ' '.join([number_words_dict.get(word, word) for word in words])
    results['9. Converter números em palavras'] = current_text

    # 10. Converter todo o texto para letras minúsculas
    current_text = current_text.lower()
    results['10. Converter para letras minúsculas'] = current_text

    # 11. Aplicar correção ortográfica
    spell = SpellChecker(language='pt')
    words = current_text.split()
    corrected_words = []
    for word in words:
        if spell.unknown([word]):
            corrected = spell.correction(word)
            if corrected is not None:
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    current_text = ' '.join(corrected_words)
    results['11. Aplicar correção ortográfica'] = current_text

    # 12. Aplicar stemização
    stemmer = nltk.stem.RSLPStemmer()
    words = current_text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    results['12. Aplicar stemização'] = ' '.join(stemmed_words)

    # 13. Aplicar lematização
    doc = nlp(current_text)
    # Manter as quebras de linha do texto original
    original_lines = text.splitlines()
    processed_lines = results['11. Aplicar correção ortográfica'].splitlines()
    lemmatized_lines = []
    for orig_line, proc_line in zip(original_lines, processed_lines):
        doc_line = nlp(proc_line)
        lemmatized_line = ' '.join([token.lemma_ for token in doc_line])
        lemmatized_lines.append(lemmatized_line)
    results['13. Aplicar lematização'] = '\n'.join(lemmatized_lines)

    return results

def process_and_save_file(file_path, output_dir, is_doril=False):
    """Lê, pré-processa e salva o resultado em um novo arquivo."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        return f"Erro: Arquivo não encontrado em {file_path}"

    results = preprocess_text(content, is_doril)

    # Criar o diretório de saída se ele não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado com sucesso.")

    # Definir o nome do arquivo de saída
    base_name = os.path.basename(file_path).split('.')[0]
    output_file_path = os.path.join(output_dir, f"{base_name}_processado.txt")

    # Escrever apenas o texto final processado no arquivo
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write(results['13. Aplicar lematização'] + "\n")

    print(f"Texto final processado salvo em: {output_file_path}")

# Caminhos para os arquivos
doril_file = 'doril.txt'
jornal_file = 'no14011801.txt'
output_folder = 'processados'

# Processar e salvar os arquivos
process_and_save_file(doril_file, output_folder, is_doril=True)
process_and_save_file(jornal_file, output_folder)