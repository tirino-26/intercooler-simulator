# Deploy Intercooler Prediction Tracker Online
# วิธี Deploy App บน Cloud (ฟรี!)

## ไฟล์ที่ต้องใช้ (มีครบแล้ว)
- `main.py` - FastAPI application
- `prediction_tracker.html` - Web interface
- `inlet_T_gas_A_lr.pkl`, `Q_A_lr.pkl`, `U_A_lr.pkl` - ML models
- `requirements.txt` - Dependencies
- `Procfile` - Start command

---

## วิธี Deploy บน Render.com (แนะนำ - ฟรี!)

### ขั้นตอน:

1. **สมัคร GitHub** (ถ้ายังไม่มี): https://github.com/join

2. **สร้าง Repository ใหม่บน GitHub:**
   - ไปที่ https://github.com/new
   - ตั้งชื่อ เช่น `intercooler-prediction`
   - กด Create repository

3. **Upload ไฟล์:**
   - กด "uploading an existing file"
   - ลากไฟล์ทั้งหมดในโฟลเดอร์นี้ไปวาง
   - กด "Commit changes"

4. **สมัคร Render.com:** https://render.com (ใช้ GitHub login)

5. **Deploy:**
   - กด "New" → "Web Service"
   - เลือก GitHub repository ที่สร้างไว้
   - ตั้งค่า:
     - Name: `intercooler-prediction`
     - Runtime: Python
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - กด "Create Web Service"

6. **รอ Deploy (~2-5 นาที)**

7. **เสร็จ!** จะได้ URL เช่น: `https://intercooler-prediction.onrender.com`

---

## วิธี Deploy บน Railway.app

1. ไปที่ https://railway.app
2. Login ด้วย GitHub
3. กด "New Project" → "Deploy from GitHub repo"
4. เลือก repository
5. Railway จะ auto-detect และ deploy ให้

---

## หลัง Deploy แล้ว

### แชร์ให้คนอื่น:
- ส่ง URL ให้เลย เช่น `https://intercooler-prediction.onrender.com`
- ทุกคนเปิดใน browser ได้เลย ไม่ต้องติดตั้งอะไร!

### ข้อดี:
- ✅ ไม่ต้องติดตั้ง Python
- ✅ ไม่ต้อง copy ไฟล์
- ✅ เข้าได้จากทุกที่ (มี internet)
- ✅ ใช้ได้ทุก device (PC, Mac, มือถือ)
- ✅ ฟรี! (Free tier)

### ข้อจำกัด Free tier:
- Render: อาจ sleep หลัง 15 นาที ไม่มีคนใช้ (จะ wake up เมื่อมีคนเข้า)
- Railway: 500 ชม./เดือน

---

## ทดสอบ Local ก่อน Deploy

```bash
cd deploy
pip install -r requirements.txt
python main.py
```

เปิด http://localhost:8000
