from fastapi import FastAPI, Form
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

@app.post("/ask")
def ask_question(question: str = Form(...), lang: str = Form('en')):
    response = process_question(question, lang)
    return {"response": response}

def preprocess_text(text, lang):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words(lang)]
    return words

def process_question(question, lang):
    if lang == 'th':
        stopwords_lang = set(stopwords.words('thai'))
    else:
        stopwords_lang = set(stopwords.words('english'))

    if "คะแนน" in question:
        return f"The faculty of information science requires a minimum score of X in {lang}."
    elif "criteria" in question:
        return f"The selection criteria include A, B, and C in {lang}. Please check the faculty's website for details."
    elif "best" in question and "branch" in question:
        return generate_response(lang)
    else:
        return f"Sorry, I cannot answer this question in {lang}."

def generate_response(lang):
    if lang == 'th':
        corpus = nltk.corpus.thai_sents()
    else:
        corpus = nltk.corpus.reuters.sents()

    flattened_corpus = [word.lower() for sent in corpus for word in sent]
    bigrams = list(nltk.ngrams(flattened_corpus, 2))
    
    start_bigram = random.choice(bigrams)
    response = list(start_bigram)
    
    for _ in range(10):
        next_word = [bigram[1] for bigram in bigrams if bigram[0] == response[-1]]
        if next_word:
            response.append(random.choice(next_word))
        else:
            break
    
    return ' '.join(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
