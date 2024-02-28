from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from pythainlp.tokenize import word_tokenize
import numpy as np
import thaispellcheck

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
        "examples": ["IT", "ITคืออะไร", "ITคืออะไรคะ", "ITคืออะไรครับ", "ITคืออะไรคะ", "ITคืออะไรครับ", "ITคืออะไรคะ", "ITคืออะไรครับ", "ITคืออะไรคะ", "ITคืออะไรครับ", "สาขา IT", "IT เรียนอะไรบ้าง"],
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



sentences = []
labels = []
for intent, intent_data in dataset.items():
    for example in intent_data["examples"]:
        sentences.append(example)
        labels.append(intent)

# Creating TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

def chatbot_response(text):
    # Transform input text
    text_vector = vectorizer.transform([text])

    # Calculate similarities
    similarities = cosine_similarity(text_vector, X)

    # Find the closest example
    closest = np.argmax(similarities, axis=1)[0]

    # Identify the intent
    intent = labels[closest]

    # Select a random response
    response = random.choice(dataset[intent]["responses"])
    return response




