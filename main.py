import nltk
from fastapi import FastAPI, Form
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import NorvigSpellChecker
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
import pythainlp
from transformers import pipeline

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
mrcpipeline = pipeline("question-answering", model="MyMild/finetune_iapp_thaiqa")
# Download necessary data
nltk.download('punkt')
nltk.download('stopwords')

# Setup for spell checker
custom_words = set(thai_words())
trie = dict_trie(dict_source=custom_words)
spell_checker = NorvigSpellChecker(custom_dict=trie)
from transformers import pipeline
from transformers import pipeline


app = FastAPI()

rule_based_responses = {
    "เกณฑ์การรับสมัคร": "เกณฑ์การรับสมัครประกอบด้วย GPA ขั้นต่ำ 3.0, การเสร็จสิ้นหลักสูตรพื้นฐาน, และการส่งคำชี้แจงส่วนบุคคล",
    "กำหนดการสมัคร": "กำหนดการสมัครสำหรับฤดูใบไม้ร่วง 2022 คือ 30 มิถุนายน 2022",
    

    "คณะวิทยาการสารสนเทศ": "มีเป็นช่วงๆค่ะ, กิจกรรมจะมีเยอะในช่วงปี 1, ซึ่งส่วนใหญ่จะเป็นกิจกรรมของมหาลัยค่ะ",
    "คณะ": "มีเป็นช่วงๆค่ะ, กิจกรรมจะมีเยอะในช่วงปี 1, ซึ่งส่วนใหญ่จะเป็นกิจกรรมของมหาลัยค่ะ",
    "กิจกรรม": "มีเป็นช่วงๆค่ะ, กิจกรรมจะมีเยอะในช่วงปี 1, ซึ่งส่วนใหญ่จะเป็นกิจกรรมของมหาลัยค่ะ",

    "สาขา IT": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",
    "IT": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",
    "ITDI": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",
    "เตรียมตัว": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",
    "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI)": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",
    "เรียน": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",
    "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": "สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI) , ต้องมีความรู้พื้นฐานเกี่ยวกับคณิตศาสตร์, หลักการโปรแกรม, ความน่าจะเป็นและสถิติสําหรับคอมพิวเตอร์, หลักการโปรแกรมเชิงวัตถุ, โครงสร้างขอมูลและอัลกอริทึม, ซึ่งรายวิชาพวกนี้เป็นรายวิชาหมวดวิชาศึกษาทั่วไป, จึงควรศึกษาไว้, เพื่อเตรียมตัวเข้าเรียนในสาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(IT), ค่ะ",

    "สาขา IT": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "IT": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "ITDI": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "ค่าเทอม": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "ค่าเทอมคณะ": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI)": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "ค่าใช้จ่ายตลอดหลักสูตร": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",
    "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": "ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ",

    "คณะวิทยาการสารสนเทศ": ["สังคมที่นี่ดีค่ะ , คนในคณะส่วนใหญ่, จะค่อนข้างช่วยเหลือกันและกัน, ทั้งเรื่องการเรียน"],
    "IT": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ"],
    "ITDI": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ"],
    "ค่าเทอม": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ"],
    "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI)": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ"],
    "ค่าใช้จ่ายตลอดหลักสูตร": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ"],
    "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ184,000บาท  , (ภาคการศึกษาละ23,000บาท), ค่ะ"],
}
con = ' '.join(''.join(value) if isinstance(value, list) else value for value in rule_based_responses.values())



@app.post("/ask")
async def ask_question(question: str = Form(...)):
    response = predefined_answer(question)
    if response:
        print("Predefined response:", response)
        return {"response": response}
    else:
        corrected_question = preprocess_text(question, lang='th')
        print("Generated response:", generate_answer(corrected_question))
        return {"response":generate_answer(corrected_question)}
    


def preprocess_text(text, lang='th'):
    if lang != 'th':
        raise ValueError("This function currently only supports Thai language.")
    words = word_tokenize(text, keep_whitespace=False)
    corrected_words = [spell_checker.correct(word) for word in words]
    return ' '.join(corrected_words)

# def generate_response(question):
#     model_name = "google/mt5-small"
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = TFT5ForConditionalGeneration.from_pretrained(model_name)
#     input_ids = tokenizer.encode(question, return_tensors="tf")
#     output = model.generate(input_ids, max_length=150, num_return_sequences=1, num_beams=5, temperature=0.7)
#     response_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     return clean_response(response_text) if clean_response(response_text).strip() != "" else "ขออภัย, ไม่สามารถตอบคำถามนี้ได้"



def generate_answer(question):
    response = mrcpipeline(question=question, context=con)
    return response['answer']



# Example usage
def predefined_answer(question):
    for keyword, response in rule_based_responses.items():
        if keyword in question:
            return response
    return None  # Move `return None` outside the loop to ensure it checks all keywords

def clean_response(response_text):
    cleaned_text = response_text.replace("<extra_id_0>", "")
    return cleaned_text

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)