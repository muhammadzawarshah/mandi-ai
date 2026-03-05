import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv


# ---------------- CONFIG ----------------
load_dotenv()

# We use standard OpenAI which is extremely fast and natively handles noise well.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Mandee AI - Lightning Fast Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- FAST TEXT CONVERTER ----------------
def process_text_strictly(raw_urdu_text: str):
    """
    GPT-4o STRICT MODE:
    1. NEVER convert names (Ali, Li, Wali, Zawar) to numbers.
    2. ONLY add zeros if 'Lakh' or 'Hazar' is explicitly spoken.
    3. FIX phonetic errors instantly.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system",
                    "content": ("""
You are an URDU ASR POST-PROCESSOR for a SABZI MANDI DATA ENTRY SYSTEM.
The input text comes from OpenAI Whisper speech recognition which might have minor phonetic errors due to Mandi background noise.
Your job is to rapidly CLEAN, CORRECT, and NORMALIZE the transcription for PERFECT DATA ENTRY.

RULES:
1. Preserve names at all costs (e.g. فتح اللہ, عطاءاللہ, علی, زوار, وسیم).
2. Fix religious auto-expansions. If someone says "علیہ السلام" in a Mandi context, it's a person's name incorrectly transcribed.
3. Convert all spoken numbers to exact digits (e.g. پچیس ہزار -> 25000, دو سو -> 200).
4. If the sentence is already meaningful and correct, do not modify it. 
5. Return ONLY the corrected Urdu text. NO JSON, NO explanations, NO quotes, NO extra words.
""")
                },
                {"role": "user", "content": f"PROCESS AND FIX: {raw_urdu_text}"}
            ],
            temperature=0
        )
        cleaned = response.choices[0].message.content.strip()
        return cleaned if cleaned else raw_urdu_text
    except Exception:
        return raw_urdu_text

# ---------------- API ENDPOINT ----------------
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    start_time = time.time()
    temp_file = f"temp_{file.filename}"

    try:
        # Rapidly save the uploaded chunk
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # 🎙 Stage 1: Whisper Transcription
        # Whisper-1 is an incredibly powerful neural net that automatically ignores background noise
        # much better than manual Python noise filtering (which slows things down and causes errors).
        with open(temp_file, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="ur",
                prompt="علی، بھائی علی، زوار، پچیس لاکھ، دس ہزار، آلو، پیاز، سبزی منڈی۔ شور شرابہ، حساب، مال"
            )

        urdu_raw = transcription.text.strip()
        print(f"Whisper Output (Took {time.time() - start_time:.2f}s): {urdu_raw}")

        if not urdu_raw:
            return {"status": "error", "message": "No audio detected or just noise."}

        # 🧠 Stage 2: Iron-Clad Logic Layer with GPT-4o
        final_result = process_text_strictly(urdu_raw)
        
        print(f"Final output (Total Time {time.time() - start_time:.2f}s): {final_result}")

        return {
            "status": "success",
            "processed_text": final_result, 
            "original_urdu": urdu_raw
        }

    except Exception as e:
        print("Error during audio processing:", e)
        return {"status": "error", "message": str(e)}

    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

