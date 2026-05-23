from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import cv2
import numpy as np
import json
import easyocr
import re
import os

app = FastAPI(title="Розумний OCR Сервер")
ocr_pool = {}

def get_ocr_reader(langs_tuple: tuple):
    if langs_tuple not in ocr_pool:
        ocr_pool[langs_tuple] = easyocr.Reader(list(langs_tuple), gpu=False)
    return ocr_pool[langs_tuple]

def clean_and_fix_text(text: str) -> str:
    # Базові автозаміни артефактів EasyOCR
    text = re.sub(r"отруііна", "отруйна", text, flags=re.IGNORECASE)
    text = re.sub(r"отруіну", "отруйну", text, flags=re.IGNORECASE)
    text = re.sub(r"отруіна", "отруйна", text, flags=re.IGNORECASE)
    text = re.sub(r"зсрл", "зерн", text, flags=re.IGNORECASE)
    text = re.sub(r"яІтлов", "житлов", text, flags=re.IGNORECASE)
    text = re.sub(r"яІтловнх", "житлових", text, flags=re.IGNORECASE)
    text = re.sub(r"Впробник", "Виробник", text, flags=re.IGNORECASE)
    
    # Видалення сміття по краях рядків
    text = re.sub(r'\b[IlH;\[\{\&Xі]{1,2}\b', '', text)
    text = re.sub(r'(8I7|iwit|IpIH|Xittii)', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...), languages: str = Form('["uk", "en"]')):
    try:
        target_langs = json.loads(languages)
        if "en" not in target_langs: target_langs.append("en")
        reader = get_ocr_reader(tuple(sorted(target_langs)))

        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        img_rgb = cv2.cvtColor(cv2.GaussianBlur(enhanced, (3, 3), 0), cv2.COLOR_GRAY2RGB)
        
        result = reader.readtext(img_rgb, paragraph=False)
        if not result: return {"status": "empty", "full_text": "", "blocks": []}

        raw_blocks = []
        for item in result:
            box, text_content = item[0], item[1].strip()
            x_min = float(min(p[0] for p in box))
            y_min = float(min(p[1] for p in box))
            x_max = float(max(p[0] for p in box))
            y_max = float(max(p[1] for p in box))
            
            cleaned = clean_and_fix_text(text_content)
            if len(cleaned) >= 2:
                raw_blocks.append({"text": cleaned, "y": y_min, "x": x_min, "h": y_max - y_min, "w": x_max - x_min})

        raw_blocks.sort(key=lambda b: b["y"])
        blocks, full_text_chunks = [], []
        
        for i, block in enumerate(raw_blocks):
            is_new_para = i > 0 and (block["y"] - (raw_blocks[i-1]["y"] + raw_blocks[i-1]["h"])) > 15.0
            display_text = block["text"]
            if is_new_para:
                display_text = "\n\n" + display_text
                full_text_chunks.append("\n\n" + block["text"])
            else:
                full_text_chunks.append(" " + block["text"] if i > 0 else block["text"])

            blocks.append({
                "text": display_text, "confidence": 1.0,
                "geometry": {"x": round(block["x"], 1), "y": round(block["y"], 1), "width": round(block["w"], 1), "height": round(block["h"], 1)}
            })

        return {"status": "success", "detected_languages": target_langs, "full_text": "".join(full_text_chunks), "blocks": blocks}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
