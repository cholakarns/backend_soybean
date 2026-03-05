from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI()

# เปิด CORS เพื่อให้ Frontend เรียกใช้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('beandetectlast.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. อ่านรูปภาพ
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. ประมวลผลด้วย YOLO
        results = model.predict(source=img, conf=0.4)
        
        # 3. นับจำนวนถั่วแต่ละชนิด
        class_ids = results[0].boxes.cls.int().tolist()
        names = model.names
        counts = {}
        for id in class_ids:
            label = names[id]
            counts[label] = counts.get(label, 0) + 1
        
        # สร้างข้อความสรุป (เช่น "ถั่วแดง: 5, ถั่วเขียว: 10")
        summary = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        message = f"พบทั้งหมด {len(class_ids)} เมล็ด ({summary})" if class_ids else "ไม่พบเมล็ดถั่วในภาพ"

        # 4. วาดผลลัพธ์ลงบนรูปและแปลงเป็น Base64
        res_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', res_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "message": message,
            "image_base64": img_base64
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
