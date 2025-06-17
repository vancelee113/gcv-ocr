import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import datetime
import re
import cv2
from PIL import Image, ImageDraw
from google.cloud import vision
import fitz  # PyMuPDF

print("=== Google Cloud Vision PDF OCR (本機安全版) ===")

# === 1) 輸入金鑰路徑，建立 .env ===
gcv_key = input("請輸入 Google Cloud JSON 金鑰路徑：").strip()
if not os.path.exists(gcv_key):
    print(f"❌ 找不到金鑰: {gcv_key}")
    exit(1)

with open(".env", "w", encoding="utf-8") as f:
    f.write(f"GOOGLE_APPLICATION_CREDENTIALS={gcv_key}\n")
print("✅ 已產生 .env")

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# === 2) 輸入 PDF 資料夾 ===
pdf_dir = input("請輸入 PDF 資料夾路徑：").strip()
if not os.path.exists(pdf_dir):
    print(f"❌ 找不到資料夾: {pdf_dir}")
    exit(1)

FIELDS = {
    "編號": (260, 650, 130, 220),
    "姓名": (390, 650, 760, 120),
    "性別": (470, 770, 90, 100),
    "出生年月日": (1120, 660, 430, 200),
    "戶籍地址": (1560, 660, 1170, 220)
}

client = vision.ImageAnnotatorClient()

def preprocess_image_default(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

def clean_text(text, field_name):
    text = text.strip().replace("\n", "")
    if field_name in ["編號", "性別"]:
        return ''.join(filter(str.isdigit, text))
    return text

def smart_gender_match(text):
    if text in ["1", "I", "L", "|", "一", "‖", "4"]: return "1"
    if text in ["2", "Z", "乙", "七", "S", "九"]: return "2"
    return "?"

def calculate_age(birth_text):
    try:
        digits = ''.join(filter(str.isdigit, birth_text))
        if len(digits) >= 2:
            birth_year = int(digits[:2]) + 1911
            age = datetime.date.today().year - birth_year
            return age
    except:
        return "?"

def extract_address_parts(address_text):
    try:
        result = {"里名": "?", "鄰": "?", "路名": "?"}
        li_positions = [(address_text.find(c), c) for c in ["里", "田", "野"] if address_text.find(c) != -1]
        if li_positions:
            idx, _ = min(li_positions)
            if idx >= 2:
                result["里名"] = address_text[idx-2:idx]
            else:
                result["里名"] = address_text[:idx]
        match_lin = re.search(r'(\d{1,3})鄰', address_text)
        if match_lin:
            result["鄰"] = match_lin.group(1)
        match_road = re.search(r'[\u4e00-\u9fa5]{2,10}(路|街)', address_text)
        if match_road:
            result["路名"] = match_road.group(0)
        return result
    except:
        return {"里名": "?", "鄰": "?", "路名": "?"}

def detect_nonempty_rows(image_cv, max_rows=6):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    row_list = []
    for i in range(max_rows):
        y_start = i * 220 + 650
        y_end = y_start + 220
        row_crop = bw[y_start:y_end, :]
        if cv2.countNonZero(row_crop) > 25000:
            row_list.append(i)
    return row_list

results = []
preview_image_shown = False

for filename in os.listdir(pdf_dir):
    if not filename.endswith(".pdf"): continue
    pdf_path = os.path.join(pdf_dir, filename)
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        row_detect = detect_nonempty_rows(image_cv)

        if page_num == 0 and not preview_image_shown:
            img = image.copy()
            draw = ImageDraw.Draw(img)
            y_offset = 220
            for row_idx in row_detect:
                for key, (x, y, w, h) in FIELDS.items():
                    ay = y + row_idx * y_offset
                    draw.rectangle([x, ay, x+w, ay+h], outline="blue", width=3)
            img.save("debug_preview.png")
            print("✅ 已產生藍框預覽圖: debug_preview.png")
            preview_image_shown = True

        y_offset = 220
        for row_idx in row_detect:
            row = {"檔名": filename, "頁碼": page_num + 1}
            for field, (x, y, w, h) in FIELDS.items():
                ay = y + row_idx * y_offset
                crop = image_cv[ay:ay+h, x:x+w]
                processed = preprocess_image_default(crop)
                _, encoded_img = cv2.imencode(".png", processed)
                vision_img = vision.Image(content=encoded_img.tobytes())
                response = client.document_text_detection(image=vision_img)
                text = response.full_text_annotation.text.strip()
                cleaned = clean_text(text, field)
                if field == "性別":
                    cleaned = smart_gender_match(cleaned)
                elif field == "出生年月日":
                    row["年齡"] = calculate_age(cleaned)
                elif field == "戶籍地址":
                    cleaned = re.sub(r'(\d{1,3})都', r'\1鄰', cleaned)
                    cleaned = cleaned.replace("類", "鄰").replace("野", "里").replace("田", "里")
                    parts = extract_address_parts(cleaned)
                    row["里名"] = parts["里名"]
                    row["鄰"] = parts["鄰"]
                    row["路名"] = parts["路名"]
                row[field] = cleaned if cleaned else "?"
            results.append(row)

df = pd.DataFrame(results)
output_xlsx = "ocr_result.xlsx"
df.to_excel(output_xlsx, index=False)
print(f"🎉 OCR 完成！結果已儲存：{output_xlsx}")