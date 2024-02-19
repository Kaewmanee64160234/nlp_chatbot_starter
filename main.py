from fastapi import FastAPI, Form

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

@app.post("/ask")
def ask_question(question: str = Form(...)):
    response = process_question(question)
    return {"response": response}

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    return words

def process_question(question):
    if "score" in question:
        return "The faculty of information science requires a minimum score of X."
    elif "criteria" in question:
        return "The selection criteria include A, B, and C. Please check the faculty's website for details."
    elif "best" in question and "branch" in question:
        return generate_response()
    else:
        return "Sorry, I cannot answer this question."

def generate_response():
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
