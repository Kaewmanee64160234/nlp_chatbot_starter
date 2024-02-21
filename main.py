from fastapi import FastAPI, Form
from pythainlp.tokenize import word_tokenize
from transformers import pipeline
from pythainlp.spell import correct
# Assuming TextBlob is imported for English auto-correction
from textblob import TextBlob

app = FastAPI()

rule_based_responses = {
    "เกณฑ์การรับสมัคร": "เกณฑ์การรับสมัครประกอบด้วย GPA ขั้นต่ำ 3.0, การเสร็จสิ้นหลักสูตรพื้นฐาน, และการส่งคำชี้แจงส่วนบุคคล",
    "กำหนดการสมัคร": "กำหนดการสมัครสำหรับฤดูใบไม้ร่วง 2022 คือ 30 มิถุนายน 2022",
}

# Cache tokenized key phrases to reduce redundant processing
tokenized_key_phrases = {key: word_tokenize(key, keep_whitespace=False) for key in rule_based_responses.keys()}

# Using a multilingual model that includes Thai language support
generator = pipeline('text-generation', model="bert-base-multilingual-cased")

def autocorrect_text(text: str, lang='th'):
    if lang == 'th':
        return correct(text)
    else:
        # For English, use TextBlob or another method for autocorrection
        corrected_text = str(TextBlob(text).correct())
        return corrected_text

def preprocess_question(question: str, lang: str):
    corrected_question = autocorrect_text(question, lang)
    tokens = word_tokenize(corrected_question, keep_whitespace=False)
    return tokens

def get_response(question: str, lang='th'):
    tokens = preprocess_question(question, lang)
    for key_phrase, response in rule_based_responses.items():
        if any(token in tokens for token in tokenized_key_phrases[key_phrase]):
            return response
    # If no rule-based response is found, return a default message
    return "ขออภัย, ฉันไม่สามารถตอบคำถามนี้ได้"

@app.post("/ask")
async def ask_question(question: str = Form(...), lang: str = Form('th')):
    response = get_response(question, lang)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
