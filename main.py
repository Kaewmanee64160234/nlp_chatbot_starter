
import random
import nltk
from fastapi import FastAPI, Form
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import NorvigSpellChecker
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
import pythainlp
from transformers import pipeline
from fastapi import FastAPI, Request, Response
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from pythainlp.tokenize import word_tokenize
import numpy as np
import thaispellcheck

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
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from fastapi.responses import JSONResponse
from starlette.status import HTTP_405_METHOD_NOT_ALLOWED
app = FastAPI()
dataset = {
    "เกณฑ์การรับสมัคร": {
        "examples": ["เกณฑ์การรับสมัคร", "กำหนดการสมัคร", "กำหนดการสมัครเป็นอย่างไรบ้าง", 'กำหนดการสมัครคิดยังไงง'],
        "responses": ["เกณฑ์การรับสมัครประกอบด้วย GPA ขั้นต่ำ 3.0, การเสร็จสิ้นหลักสูตรพื้นฐาน, และการส่งคำชี้แจงส่วนบุคคล"]
    },
    "สาขา": {
        "examples": ["คณะนี้มีสาขาอะไรบ้าง", "คณะนี้มีกี่สาขา", "คณะนี้มีสาขาอะไรบ้าง", "เกณฑ์การรับสมัคร"],
        "responses": ["วิทยาการคอมพิวเตอร์ (Computer Science: CS), เป็นศาสตร์เกี่ยวกับการศึกษาค้นคว้าทฤษฏีการคำนวณสำหรับคอมพิวเตอร์, และทฤษฏีการประมวลผลสารสนเทศ, ทั้งด้านซอฟต์แวร์, ฮาร์ดแวร์, และเครือข่าย, ประกอบด้วยหลายหัวข้อที่เกี่ยวข้อง, เช่น, การวิเคราะห์และสังเคราะห์ขั้นตอนวิธี, ทฤษฎีภาษาโปรแกรม, ทฤษฏีการพัฒนาซอฟต์แวร์, ทฤษฎีฮาร์ดแวร์คอมพิวเตอร์, และทฤษฏีเครือข่าย, เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล (Information Technology for Digital Industry : ITDI), เป็นศาสตร์เกี่ยวกับการประยุกต์ใช้เทคโนโลยีในการประมวลผลสารสนเทศ, ซึ่งครอบคลุมถึงการรับ-ส่ง, การแปลง, การจัดเก็บ, การประมวลผล, และการค้นคืนสารสนเทศ, เป็นการประยุกต์ใช้ทฤษฎีและขั้นตอนวิธีจากวิทยาการคอมพิวเตอร์ในการทำงาน, การศึกษาอุปกรณ์ต่างๆทางเทคโนโลยีสารสนเทศ, การวางโครงสร้างสถาปัตยกรรมองค์กรด้วยเทคโนโลยีสารสนเทศอย่างมีประสิทธิภาพสูงสุดกับสังคม ธุรกิจ องค์กร หรืออุตสาหกรรม, 	วิศวกรรมซอฟต์แวร์ (Software Engineering: SE), เป็นศาสตร์เกี่ยวกับวิศวกรรมด้านซอฟต์แวร์, เกี่ยวข้องกับการใช้กระบวนการทางวิศวกรรมในการดูแลการผลิตซอฟต์แวร์ที่สามารถปฏิบัติงานตามเป้าหมาย, ภายใต้เงื่อนไขที่กำหนด, โดยเริ่มตั้งแต่การเก็บความต้องการ, การตั้งเป้าหมายของระบบ, การออกแบบ, กระบวนการพัฒนา, การตรวจสอบ, การประเมินผล, การติดตามโครงการ, การประเมินต้นทุน, การรักษาความปลอดภัย, ไปจนถึงการคิดราคาซอฟต์แวร์, เป็นต้น, ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (Applied Artificial Intelligence and Smart Technology: AAI), ได้พัฒนาขึ้นเพื่อตอบสนองต่อความต้องการบุคลากรด้านเทคโนโลยีสารสนเทศเพื่อการเปลี่ยนรูปองค์การไปสู่องค์กรอัจฉริยะที่ขับเคลื่อนด้วยข้อมูล (Data-driven Business) บนพื้นฐานของเทคโนโลยีปัญญาประดิษฐ์, ตลอดถึงการพัฒนากำลังคนในธุรกิจดิจิทัล, และระบบอัจฉริยะ, เช่น, โรงงานอัจฉริยะ (Smart Factory), เกษตรอัจฉริยะ (Smart Agriculture), ฟาร์มอัจฉริยะ (Smart Farming), เมืองอัจฉริยะ (Smart City), การบริการอัจฉริยะ (Smart Services), การท่องเที่ยวอัจฉริยะ (Smart Tourisms), และโลจิสติกส์อัจฉริยะ (Smart Logistrics), สอดคล้องกับโครงการเขตพัฒนาพิเศษภาคตะวันออก (EEC) ภายใต้แผนยุทธศาสตร์ประเทศไทย 4.0"]
    },
    "กำหนดการสมัคร": {
        "examples": ["กำหนดการสมัครวันไหน", "สมัครเข้าเรียนได้ตอนไหน", "สามารถสมัครได้ตอนไหน"],
        "responses": ["กำหนดการสมัครสำหรับฤดูใบไม้ร่วง 2022 คือ 30 มิถุนายน 2022"]
    },
    "ค่าเรียน": {
        "examples": ["สาขา ITค่าทอมเท่าไหร่", "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัลค่าเทอมเท่าไหร่", "ค่าใช้จ่ายตลอดหลักสูตร", "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล(ITDI)", "ค่าเทอมเท่าไหร่", "ITDIค่าทอมเท่าไหร่", "ITค่าทอมเท่าไหร่", "ค่าเรียนเท่าไหร่", "ค่าเรียนเท่าไหร่บ้าง", "ค่าเรียนเท่าไหร่ครับ", "ค่าเรียนเท่าไหร่คะ"],
        "responses": ["ค่าใช้จ่ายตลอดหลักสูตรโดยประมาณ 184,000บาท (ภาคการศึกษาละ 23,000 บาท) ค่ะ"]
    },
    "สอบเข้า": {
        "examples": ["สอบเข้าเมื่อไหร่", "สอบเข้าวันไหน", "สอบเข้าเมื่อไหร่ครับ", "สอบเข้าเมื่อไหร่คะ", "สอบเข้าวันไหนคะ", "สอบเข้าวันไหนครับ"],
        "responses": ["สอบเข้าวันที่ 9 ก.ย. 2022"]
    },
    "สมัครเรียน": {
        "examples": ["สมัครเรียนได้ที่ไหน", "สมัครเรียนอย่างไร", "สมัครเรียนที่ไหน"],
        "responses": ["สามารถสมัครเรียนได้ที่ https://reg.buu.ac.th/registrar/home.asp/"]
    },
    "สภาพแวดล้อมและสังคม": {
        "examples": ["สภาพแวดล้อมและสังคม", "สภาพแวดล้อม", "สภาพสังคม", "สภาพแวดล้อมและสังคมเป็นอย่างไร", "สภาพแวดล้อมและสังคมเป็นอย่างไรบ้าง", "สังคมเป็นยังไง"],
        "responses": ["สังคมที่นี่ดีค่ะ , คนในคณะส่วนใหญ่, จะค่อนข้างช่วยเหลือกันและกัน, ทั้งเรื่องการเรียน, การใช้ชีวิต"]
    },
    "อุปกรณ์การเรียน": {
        "examples": ["อุปกรณ์การเรียน", "อุปกรณ์การเรียนมีอะไรบ้าง", "อุปกรณ์การเรียนมีอะไรบ้างคะ", "อุปกรณ์การเรียนมีอะไรบ้างครับ", "อุปกรณ์การเรียนมีอะไรบ้างครับ", "อุปกรณ์การเรียนมีอะไรบ้างคะ", "อุปกรณ์การเรียนไม่มีได้ไหม", "ไม่มีอุปกรณ์การเรียนยืมได้มั้ย", "อุปกรณ์มีไหม", "มีอุปกรณ์มั้ย"],
        "responses": ["อุปกรณ์การเรียนที่จำเป็นคือ คอมพิวเตอร์ โน๊ตบุ๊ค หรืออุปกรณ์ที่สามารถเข้าถึงอินเทอร์เน็ตได้ หากไม่มีทางคณะมีอุปกรณ์ให้ยืมกลับบ้านได้ค่ะ"]
    },
    "การเรียน": {
        "examples": ["การเรียน", "การเรียนเป็นยังไง", "การเรียนเป็นยังไงบ้าง", "การเรียนเป็นยังไงคะ", "การเรียนเป็นยังไงครับ", "การเรียนเป็นยังไงคะ", "การเรียนเป็นยังไงครับ", "การเรียนเป็นยังไงคะ", "เรียนยากมั้ย", "คณะนี้เรียนยากมั้ย"],
        "responses": ["การเรียนที่นี่ดีค่ะ มีอาจารย์ที่ชำนาญในสาขาวิชา และมีการสอนที่ดี มีการสนับสนุนในการเรียนอย่างดี มีการสนับสนุนในการเรียนอย่างดี", "เป็นคำถามที่น่าสนใจนะคะ , จะเรียกว่ายากก็อาจจะยาก, จะเรียกว่าง่ายก็อาจจะง่าย, อยู่ที่ความชอบและความถนัดของแต่ละคน, ถ้าเราชอบในด้านนี้และสนุกกับการเรียนรู้เกี่ยวกับสายไอที, ก็จะมองว่ามันง่ายและสนุกไปกับการเรียน, แต่สำหรับคนที่ไม่ชอบหรือสนใจแต่ไม่เก่งด้านนี้, ก็อาจจะยากแต่ไม่ยากเกินไปแน่นอนค่ะ"]
    },
    "SE": {
        "examples": ["SE", "SEคืออะไร", "SEคืออะไรคะ", "SEคืออะไรครับ", "SEคืออะไรคะ", "SEคืออะไรครับ", "SEคืออะไรคะ", "SEคืออะไรครับ", "SEคืออะไรคะ", "SEคืออะไรครับ", "สาขา SE", "SE"],
        "responses": ["วิศวกรรมซอฟต์แวร์ (Software Engineering: SE), เป็นศาสตร์เกี่ยวกับวิศวกรรมด้านซอฟต์แวร์, เกี่ยวข้องกับการใช้กระบวนการทางวิศวกรรมในการดูแลการผลิตซอฟต์แวร์ที่สามารถปฏิบัติงานตามเป้าหมาย, ภายใต้เงื่อนไขที่กำหนด, โดยเริ่มตั้งแต่การเก็บความต้อการ, การตั้งเป้าหมายของระบบ, การออกแบบ, กระบวนการพัฒนา, การตรวจสอบ, การประเมินผล, การติดตามโครงการ, การประเมินต้นทุน, การรักษาความปลอดภัย, ไปจนถึงการคิดราคาซอฟต์แวร์, เป็นต้น"]
    },
    "CS": {
        "examples": ["CS", "CSคืออะไร", "CSคืออะไรคะ", "CSคืออะไรครับ", "CSคืออะไรคะ", "CSคืออะไรครับ", "CSคืออะไรคะ", "CSคืออะไรครับ", "CSคืออะไรคะ", "CSคืออะไรครับ", "สาขา CS", "CS เรียนอะไรบ้าง", ""],
        "responses": ["วิทยาการคอมพิวเตอร์ (Computer Science: CS), เป็นศาสตร์เกี่ยวกับการศึกษาค้นคว้าทฤษฏีการคำนวณสำหรับคอมพิวเตอร์, และทฤษฏีการประมวลผลสารสนเทศ, ทั้งด้านซอฟต์แวร์, ฮาร์ดแวร์, และเครือข่าย, ประกอบด้วยหลายหัวข้อที่เกี่ยวข้อง, เช่น, การวิเคราะห์และสังเคราะห์ขั้นตอนวิธี, ทฤษฏีภาษาโปรแกรม, ทฤษฏีการพัฒนาซอฟต์แวร์, ทฤษฎีฮาร์ดแวร์คอมพิวเตอร์, และทฤษฏีเครือข่าย"]
    },
    "IT": {
        "examples": ["IT","ITDI", "ITคืออะไร", "ITคืออะไรคะ", "ITคืออะไรครับ", "ITคืออะไรคะ", "ITคืออะไรครับ", "ITคืออะไรคะ", "ITคืออะไรครับ", "ITคืออะไรคะ", "ITคืออะไรครับ", "สาขา IT", "IT เรียนอะไรบ้าง"],
        "responses": ["เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล (Information Technology for Digital Industry : ITDI), เป็นศาสตร์เกี่ยวกับการประยุกต์ใช้เทคโนโลยีในการประมวลผลสารสนเทศ, ซึ่งครอบคลุมถึงการรับ-ส่ง, การแปลง, การจัดเก็บ, การประมวลผล, และการค้นคืนสารสนเทศ, เป็นการประยุกต์ใช้ทฤษฎีและขั้นตอนวิธีจากวิทยาการคอมพิวเตอร์ในการทำงาน, การศึกษาอุปกรณ์ต่างๆทางเทคโนโลยีสารสนเทศ, การวางโครงสร้างสถาปัตยกรรมองค์กรด้วยเทคโนโลยีสารสนเทศอย่างมีประสิทธิภาพสูงสุดกับสังคม ธุรกิจ องค์กร หรืออุตสาหกรรม"]

    },
    "AAI": {
        "examples": ["AAI", "AAIคืออะไร", "AAIคืออะไรคะ", "AAIคืออะไรครับ", "AAIคืออะไรคะ", "AAIคืออะไรครับ", "AAIคืออะไรคะ", "AAIคืออะไรครับ", "AAIคืออะไรคะ", "AAIคืออะไรครับ", "สาขา AAI", "AAI เรียนอะไรบ้าง", "AI"],
        "responses": ["ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (Applied Artificial Intelligence and Smart Technology: AAI), ได้พัฒนาขึ้นเพื่อตอบสนองต่อความต้องการบุคลากรด้านเทคโนโลยีสารสนเทศเพื่อการเปลี่ยนรูปองค์การไปสู่องค์กรอัจฉริยะที่ขับเคลื่อนด้วยข้อมูล (Data-driven Business) บนพื้นฐานของเทคโนโลยีปัญญาประดิษฐ์, ตลอดถึงการพัฒนากำลังคนในธุรกิจดิจิทัล, และระบบอัจฉริยะ, เช่น, โรงงานอัจฉริยะ (Smart Factory), เกษตรอัจฉริยะ (Smart Agriculture), ฟาร์มอัจฉริยะ (Smart Farming), เมืองอัจฉริยะ (Smart City), การบริการอัจฉริยะ (Smart Services), การท่องเที่ยวอัจฉริยะ (Smart Tourisms), และโลจิสติกส์อัจฉริยะ (Smart Logistrics), สอดคล้องกับโครงการเขตพัฒนาพิเศษภาคตะวันออก (EEC) ภายใต้แผนยุทธศาสตร์ประเทศไทย 4.0"]
    },
    "เรียนต่อ": {
        "examples": ["เรียนต่อ", "เรียนต่อยังไง", "เรียนต่อยังไงบ้าง", "เรียนต่อยังไงคะ", "เรียนต่อยังไงครับ", "เรียนต่อยังไงคะ", "เรียนต่อยังไงครับ", "เรียนต่อยังไงคะ", "เรียนต่อยังไงครับ", "เรียนต่อยังไงคะ", "เรียนต่อยังไงครับ"],
        "responses": ["เรียนต่อที่นี่ดีค่ะ มีความร่วมมือในการเรียนรู้ มีอาจารย์ที่ชำนาญในสาขาวิชา และมีการสนับสนุนในการเรียนอย่างดี มีการสนับสนุนในการเรียนอย่างดี ในระดับปริญญาเอก  , ทางคณะวิทยาการสารสนเทศ  , มหาวิทยาลัยบูรพา  , เปิดสอนหลักสูตร , ปรัชญาดุษฎีบัณฑิต  , สาขาวิชาวิทยาการข้อมู"]
    },
    "เรียนต่อที่ไหน": {
        "examples": ["เรียนต่อที่ไหน", "เรียนต่อที่ไหนดี", "เรียนต่อที่ไหนดีคะ", "เรียนต่อที่ไหนดีครับ", "เรียนต่อที่ไหนดีคะ", "เรียนต่อที่ไหนดีครับ", "เรียนต่อที่ไหนดีคะ", "เรียนต่อที่ไหนดีครับ", "เรียนต่อที่ไหนดีคะ", "เรียนต่อที่ไหนดีครับ"],
        "responses": ["เรียนต่อที่นี่ดีค่ะ มีความร่วมมือในการเรียนรู้ มีอาจารย์ที่ชำนาญในสาขาวิชา และมีการสนับสนุนในการเรียนอย่างดี มีการสนับสนุนในการเรียนอย่างดี ในระดับปริญญาเอก  , ทางคณะวิทยาการสารสนเทศ  , มหาวิทยาลัยบูรพา  , เปิดสอนหลักสูตร , ปรัชญาดุษฎีบัณฑิต  , สาขาวิชาวิทยาการข้อมูล"]
    },
    "อาจารย์แพร": {
        "examples": ["รู้จักอาจารย์แพรมั้ย", "อาจารย์แพร", "อาจารย์แพรเป็นยังไง", "อาจารย์แพรเป็นยังไงบ้าง", "อาจารย์แพรเป็นยังไงคะ", "อาจารย์แพรเป็นยังไงครับ", "อาจารย์แพรเป็นยังไงคะ", "อาจารย์แพรเป็นยังไงครับ", "อาจารย์แพรเป็นยังไงคะ", "อาจารย์แพรเป็นยังไงครับ", "อาจารย์แพรเป็นยังไงคะ", "อาจารย์แพรเป็นยังไงครับ"],
        "responses": ["อาจารย์แพรเป็นอาจารย์ที่น่ารักค่ะ อาจารย์แพรเป็นอาจารย์ที่ชำนาญในสาขาวิชา และมีการสนับสนุนในการเรียนอย่างดี มีการสนับสนุนในการเรียนอย่างดี"]
    },
    "อาจรย์กบ": {
        "examples": ["รู้จักอาจารย์กบมั้ย", "อาจารย์กบ", "อาจารย์กบเป็นยังไง", "อาจารย์กบเป็นยังไงบ้าง", "อาจารย์กบเป็นยังไงคะ", "อาจารย์กบเป็นยังไงครับ", "อาจารย์กบเป็นยังไงคะ", "อาจารย์กบเป็นยังไงครับ", "อาจารย์กบเป็นยังไงคะ", "อาจารย์กบเป็นยังไงครับ", "อาจารย์กบเป็นยังไงคะ", "อาจารย์กบเป็นยังไงครับ"],
        "responses": ["อ๊บ อ๊บ"]
    },
    "ความแตกต่างระหว่างสาขา": {
        "examples": ["ความแตกต่างระหว่างสาขา", "ความแตกต่างระหว่างสาขาเป็นยังไง", "ความแตกต่างระหว่างสาขาเป็นยังไงบ้าง", "ความแตกต่างระหว่างสาขาเป็นยังไงคะ", "ความแตกต่างระหว่างสาขาเป็นยังไงครับ", "ความแตกต่างระหว่างสาขาเป็นยังไงคะ", "ความแตกต่างระหว่างสาขาเป็นยังไงครับ", "ความแตกต่างระหว่างสาขาเป็นยังไงคะ", "ความแตกต่างระหว่างสาขาเป็นยังไงครับ", "ความแตกต่างระหว่างสาขาเป็นยังไงคะ", "ความแตกต่างระหว่างสาขาเป็นยังไงครับ"],
        "responses": ["ความแตกต่างระหว่างสาขา คือ วิทยาการคอมพิวเตอร์ (Computer Science: CS), เป็นศาสตร์เกี่ยวกับการศึกษาค้นคว้าทฤษฏีการคำนวณสำหรับคอมพิวเตอร์, และทฤษฏีการประมวลผลสารสนเทศ, ทั้งด้านซอฟต์แวร์, ฮาร์ดแวร์, และเครือข่าย, ประกอบด้วยหลายหัวข้อที่เกี่ยวข้อง, เช่น, การวิเคราะห์และสังเคราะห์ขั้นตอนวิธี, ทฤษฏีภาษาโปรแกรม, ทฤษฏีการพัฒนาซอฟต์แวร์, ทฤษฎีฮาร์ดแวร์คอมพิวเตอร์, และทฤษฏีเครือข่าย, เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล (Information Technology for Digital Industry : ITDI), เป็นศาสตร์เกี่ยวกับการประยุกต์ใช้เทคโนโลยีในการประมวลผลสารสนเทศ, ซึ่งครอบคลุมถึงการรับ-ส่ง, การแปลง, การจัดเก็บ, การประมวลผล, และการค้นคืนสารสนเทศ, เป็นการประยุกต์ใช้ทฤษฎีแะขั้นตอนวิธีจากวิทยาการคอมพิวเตอร์ในการทำงาน, การศึกษาอุปกรณ์ต่างๆทางเทคโนโลยีสารสนเทศ, การวางโครงสร้างสถาปัตยกรรมองค์กรด้วยเทคโนโลยีสารสนเทศอย่างมีประสิทธิภาพสูงสุดกับสังคม ธุรกิจ องค์กร หรืออุตสาหกรรม ในสาขา SE  , จะเป็นการเรียน , ที่เกี่ยวข้องกับการใช้กระบวนการทางวิศวกรรม , ในการดูแลการผลิตซอฟต์แวร์ที่สามารถปฏิบัติงานตามเป้าหมาย , ภายใต้เงื่อนไขที่กำหนด , โดยเริ่มตั้งแต่การเก็บความต้องการ, การตั้งเป้าหมายของระบบ , การออกแบบ  ,  กระบวนการพัฒนา , การตรวจสอบ , การประเมินผล , การติดตามโครงการ , การประเมินต้นทุน  , การรักษาความปลอดภัย ,  ไปจนถึงการคิดราคาซอฟต์แวร์ จะเป็นการเรียน , ที่เกี่ยวข้องกับปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ  , ได้พัฒนาขึ้นเพื่อตอบสนอง , ต่อความต้องการบุคลากร  , ด้านเทคโนโลยีสารสนเทศเพื่อการเปลี่ยนรูปองค์การไปสู่องค์กรอัจฉริยะ , ที่ขับเคลื่อนด้วยข้อมูล (Data-driven Business)  , บนพื้นฐานของเทคโนโลยีปัญญาประดิษฐ์  , ตลอดถึงการพัฒนากำลังคนในธุรกิจดิจิทัล , และระบบอัจฉริยะ  , เช่น  , โรงงานอัจฉริยะ (Smart Factory) , เกษตรอัจฉริยะ (Smart Agriculture)  , ฟาร์มอัจฉริยะ (Smart Farming)  , มืองอัจฉริยะ (Smart City)"]

    },
    "ฝึกงาน": {
        "examples": ["ฝึกงาน", "ฝึกงานยังไง", "มีฝึกงานมั้ย", "ระยะเวลาการฝึกงาน", "ช่วงการฝึกงาน"],
        "responses": ["สามารถฝึกงานได้ที่บริษัทต่างๆ ที่เกี่ยวข้องกับสาขาวิชาที่เรียนอยู่ ค่ะ"]
    },
}
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

line_bot_api = LineBotApi('W8wWPV62rWD9TEBEzZ0l3ZLVKXR3JZ5XhVMGMekvgG+9J2OgGxDQCBTj1ok/raQm44+BdgrL/vMjC193Mx4Qn8qfVpe98av9c3rFtRQ5vpiQ5XRIxfYyamR9FyC9EUz1XSOgGXRHQK6DbKkQWtONWAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('9807cfec5558c0338f2853f050eb1667')
@app.get("/")  # Handles GET requests for verification
async def verify_line_webhook(request: Request):
    print(request)
    # LINE sends a verification challenge as a query parameter
    challenge = request.query_params.get("hub.challenge")
    if challenge:
        # Echo the challenge back in the response
        return JSONResponse(content={"challenge": challenge}, status_code=200)
    else:
        # No challenge found, return an appropriate response
        return JSONResponse(content={"message": "Challenge not found"}, status_code=400)

@app.post("/")  # Your actual webhook handling POST requests
async def handle_webhook(request: Request):
    # Your webhook handling logic here
    return JSONResponse(content={"message": "Webhook received"}, status_code=200)
@app.post("/webhook")
async def line_webhook(request: Request):
    # Get request body as text
    body = await request.body()
    body_text = body.decode("utf-8")

    # Get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # Handle the webhook body
    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        return {"error": "Invalid signature."}
    except LineBotApiError as e:
        return {"error": str(e)}
    return "OK"

# Define how to handle text messages
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_message = event.message.text

    # Here, call your chatbot logic and get the response
    response = predefined_answer(user_message)  # This is your chatbot function

    # Respond to the user
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response)
    )

sentences = [example for intent_data in dataset.values() for example in intent_data["examples"]]
labels = [intent for intent, intent_data in dataset.items() for _ in intent_data["examples"]]

for intent, intent_data in dataset.items():
    for example in intent_data["examples"]:
        sentences.append(example)
        labels.append(intent)

# Creating TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
def chatbot_response(text):
    # Vectorize the input text
    text_vector = vectorizer.transform([text])
    # Calculate similarities with the dataset
    similarities = cosine_similarity(text_vector, X)
    max_similarity = np.max(similarities)
    
    # Define a threshold for determining a match (you may need to adjust this value)
    similarity_threshold = 0.1  # Example threshold
    
    if max_similarity < similarity_threshold:
        # If similarity is below threshold, return "I don't understand"
        return "ขออภัยค่ะ ฉันไม่เข้าใจคำถามของคุณ กรุณาถามคำถามใหม่อีกครั้งค่ะ"
    
    # If above threshold, find the closest matching intent
    closest = np.argmax(similarities, axis=1)[0]
    intent = labels[closest]
    response = random.choice(dataset[intent]["responses"])
    return response


def preprocess_text(text, lang='th'):
    if lang != 'th':
        raise ValueError("This function currently only supports Thai language.")
    words = word_tokenize(text, keep_whitespace=False)
    corrected_words = [spell_checker.correct(word) for word in words]
    return ' '.join(corrected_words)


def generate_answer(question):
    response = mrcpipeline(question=question, context=con)
    return response['answer']



# Example usage
def predefined_answer(question):
    tokens_newmm = word_tokenize(question)
    user_input = ' '.join(tokens_newmm)
    response = chatbot_response(user_input)
    if response:  # Checks if `response` is not None or an empty string
        return response
    else:
        return None  # Returns None if no satisfactory answer can be generated


def clean_response(response_text):
    cleaned_text = response_text.replace("<extra_id_0>", "")
    return cleaned_text

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)