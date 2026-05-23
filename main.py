from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import cv2
import numpy as np
import json
import easyocr
import re
import os

app = FastAPI(title="Розумний OCR Сервер - Повна Сумісність")
ocr_pool = {}

def get_ocr_reader(langs_tuple: tuple):
    if langs_tuple not in ocr_pool:
        print(f" Ініціалізація моделей для мов: {list(langs_tuple)}")
        ocr_pool[langs_tuple] = easyocr.Reader(list(langs_tuple), gpu=False)
    return ocr_pool[langs_tuple]

def preprocess_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w < 1600:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    final_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    return final_img, float(w * 2 / w if w < 1600 else 1.0)

def clean_and_fix_text(text: str) -> str:
    # Словник інтелектуальної заміни артефактів EasyOCR на чисту українську мову
    replacements = {
        "отруііна": "отруйна", "отруіну": "отруйну", "отруіна": "отруйна",
        "зсрл": "зерн", "Зсрн": "Зерн", "Зсрк": "Зерн", "Зсрнову": "Зернову",
        "прізнач": "признач", "прзнач": "признач", "зннщ": "знищ", 
        "мншоподіб": "мишоподіб", "мншопоіб": "мишоподіб",
        "приміш": "приміщ", "прнмішсннях": "приміщеннях",
        "яІтлов": "житлов", "ЖІІТЛОВ": "ЖИТЛОВ", "яІтловнх": "житлових",
        "пкладають": "підкладають", "ЛОЖКН": "ЛОЖКИ", "нсглибоку": "неглибоку",
        "ГрІІЗут": "Гризуни", "ГрІзут": "Гризуни", "лочинають": "починають", 
        "Збріга": "Зберіга", "Збрігали": "Зберігати", "дия дітей": "для дітей", 
        "дітейі": "дітей", "орпгінал": "оригінал", "орпгінальній": "оригінальній",
        "окремб": "окремо", "окреNo": "окремо", "віл 21": "від 21", 
        "Впробник": "Виробник
