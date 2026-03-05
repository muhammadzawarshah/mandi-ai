import os
import subprocess
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()
# OpenAI for Whisper ASR
whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenRouter for Text Processing
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
)

app = FastAPI(title="Mandee AI - Zero-Error Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- THE IRON-CLAD CONVERTER ----------------
def process_text_strictly(raw_urdu_text: str, field_type: str = "text"):
    """
    GPT-4o STRICT MODE:
    1. NEVER convert names (Ali, Li, Wali, Zawar) to numbers.
    2. ONLY add zeros if 'Lakh' or 'Hazar' is explicitly spoken.
    3. FIX common phonetic errors instantly.
    4. Force raw numbers if field_type is "number".
    """
    try:
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system",
                    "content": ("""
You are an URDU ASR POST-PROCESSOR and DATA EXTRACTOR for a SABZI MANDI DATA ENTRY SYSTEM.

IMPORTANT CONTEXT:
The input text comes from OpenAI Whisper speech recognition.
Whisper output may contain phonetic mistakes and automatic expansions.

Your job is to CLEAN, CORRECT, and NORMALIZE the transcription
for PERFECT DATA ENTRY while preserving correctness.

==================================================
CORE PRINCIPLE — PRESERVE BEFORE CORRECT
==================================================
Whisper output is assumed CORRECT by default.

ONLY modify text when there is STRONG evidence
that speech recognition made an error.

If text already forms valid Urdu → KEEP IT.

Accuracy is more important than intelligence.

==================================================
RULE 1 — WHISPER AWARENESS
==================================================
This is speech recognition output.

Possible Whisper errors:
• similar sounding words
• religious auto-expansion
• spelling distortions
• broken phonetics

You MAY fix ASR mistakes intelligently.

==================================================
RULE 2 — CONSERVATIVE SMART GUESSING
==================================================
Guess corrections ONLY when:

✓ phrase sounds unnatural
✓ phrase unlikely in mandi/business context
✓ known ASR mistake pattern exists

If unsure → DO NOT CHANGE.

==================================================
RULE 3 — ABSOLUTE NAME PROTECTION (CRITICAL)
==================================================
Person names have highest priority.

If input already contains a valid Urdu name:
NEVER replace it with another name.

Allowed:
wrong word → correct name

Forbidden:
correct name → different name

Examples (KEEP UNCHANGED):
فتح اللہ
عطاءاللہ
علی
زوار
احمد
وسیم

Never modify valid names.

==================================================
RULE 4 — RELIGIOUS PHRASE ASR FIX
==================================================
Whisper may incorrectly insert religious phrases.

If they appear in mandi/business context,
convert them into most probable spoken name.

Possible ASR expansions:
• علیہ السلام
• علیہ سلام
• رضی اللہ عنہ
• صلی اللہ علیہ وسلم
• رحمتہ اللہ علیہ

Example:
"بھائی علیہ السلام"
→ likely a person name → correct intelligently.

==================================================
RULE 5 — DOMAIN CONTEXT
==================================================
Domain is SABZI MANDI DATA ENTRY.

Prefer interpreting words as:
• person names
• vegetables
• quantities
• prices

Avoid religious interpretation unless clearly intentional.

==================================================
RULE 6 — COMPLETE NUMBER NORMALIZATION (VERY CRITICAL)
==================================================
ALL spoken Urdu numbers MUST be converted into DIGITS.

Numbers must ALWAYS appear in numeric format
for database entry.

Convert Urdu number words into digits.

Examples:

صفر → 0
ایک → 1
دو → 2
تین → 3
چار → 4
پانچ → 5
دس → 10
گیارہ → 11                       
چودہ → 14
ونیس → 19
بیس → 20
پچیس → 25
پچاس → 50
                                
انتالیس / اونتالیس → 39
انچاس → 49
اڑتالیس → 48
چالیس → 40
اکتالیس → 41
بیالیس → 42
تینتالیس → 43
چوالیس → 44
پینتالیس → 45
چھیالیس → 46
سینتالیس → 47
                                
سو → 100
ڈیڑھ سو → 150
دو سو → 200
ایک ہزار → 1000
دو ہزار → 2000
پچیس ہزار → 25000
ایک لاکھ → 100000
پچیس لاکھ → 2500000

RULES:

1. ALWAYS convert Urdu number words into digits.
2. Remove Urdu numeric spelling after conversion.
3. NEVER invent numbers.
4. If digits already exist → keep unchanged.
5. لاکھ = ×100000
6. ہزار = ×1000

==================================================
RULE 7 — NO OVER-CORRECTION
==================================================
If sentence already meaningful Urdu:
DO NOTHING.

Never modify correct text.

==================================================
RULE 8 — INTERNAL DECISION CHECK
==================================================
Before changing anything, internally verify:

1. Is original valid Urdu? → YES → KEEP
2. Is correction clearly better? → YES → CHANGE
3. Unsure? → KEEP ORIGINAL

When uncertain → PRESERVE ORIGINAL.

==================================================
FINAL OUTPUT FORMAT
==================================================
Return ONLY a valid JSON object containing the extracted data. Do NOT wrap it in markdown code blocks like ```json ... ```.

Use the following JSON schema:
{
  "customer_name": "String (extracted name if any)",
  "items": [
    {
      "item_name": "String (e.g., آلو, پیاز, ٹماٹر. MUST capture even if price/qty is missing)",
      "quantity": "Number (extracted quantity digits only, or null if missing)",
      "price": "Number (extracted price digits only, or null if missing)"
    }
  ],
  "total_amount": "Number (if a total amount is explicitly or implicitly mentioned)",
  "corrected_text": "String (the full clean corrected Urdu text)"
}

NO explanations. NO markdown. ONLY raw JSON.
"""
                    )
                },
                {"role": "user", "content": f"PROCESS AND FIX: '{raw_urdu_text}'\nTarget Field Type: {field_type}"}
            ],
            temperature=0  # ZERO temperature for maximum consistency
        )
        cleaned = response.choices[0].message.content.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"customer_name": "", "items": [], "total_amount": 0, "corrected_text": cleaned}
    except Exception:
        return raw_urdu_text

# ---------------- API ENDPOINT ----------------

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    field_type: str = Form("text")
):

    temp_file = os.path.join(tempfile.gettempdir(), f"temp_{os.urandom(8).hex()}_{file.filename}")

    cleaned_audio_file = temp_file + "_cleaned.wav"

    try:
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # 🧹 Stage 0.5: Audio Noise Reduction using FFmpeg
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_file, 
                "-af", "afftdn=nf=-25", 
                cleaned_audio_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            audio_to_process = cleaned_audio_file
        except Exception as e:
            print(f"FFmpeg noise reduction failed: {e}")
            audio_to_process = temp_file

        # 🎙 Stage 1: Whisper Transcription
        with open(audio_to_process, "rb") as audio:
            transcription = whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="ur",
                prompt="علی، بھائی علی، زوار، پچیس لاکھ، دس ہزار، آلو، پیاز، سبزی منڈی۔"
            )

        urdu_raw = transcription.text.strip()

        if not urdu_raw:
            return {"status": "error", "message": "No audio detected"}

        # 🧠 Stage 2: Iron-Clad Logic Layer
        final_result = process_text_strictly(urdu_raw, field_type)

        return {
            "status": "success",
            "processed_text": final_result, 
            "original_urdu": urdu_raw
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if 'cleaned_audio_file' in locals() and os.path.exists(cleaned_audio_file):
            os.remove(cleaned_audio_file)
