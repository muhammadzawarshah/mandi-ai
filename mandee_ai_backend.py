import os
import time
import json
import re
import subprocess
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

# ================================================================
#  MANDEE AI — v3.1 FIXED EDITION
#
#  Fixes in v3.1:
#    ✅ FFmpeg Windows auto-detect (WinError 2 fixed)
#    ✅ Gemini 3.1 Pro JSON parse fixed (no response_format)
#    ✅ Much stronger Urdu prompt — galat likhna band
#    ✅ Faster pipeline — Gemini Flash as PRIMARY (fastest)
#    ✅ Better number + fraction handling
#
#  LLM ORDER (speed + accuracy balanced):
#    TIER 1: google/gemini-2.5-flash        ← Fastest, very good Urdu
#    TIER 2: anthropic/claude-opus-4.6      ← Best accuracy fallback
#    TIER 3: google/gemini-3.1-pro-preview  ← Last resort
# ================================================================

load_dotenv()

whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

app = FastAPI(title="Mandee AI v3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# FFMPEG — Windows + Linux/Mac auto-detect
# ================================================================
def find_ffmpeg() -> str | None:
    """
    FFmpeg binary dhundo — Windows aur Linux dono pe kaam kare.
    Common Windows install locations check karta hai.
    """
    # 1. PATH mein check karo (sab se pehle)
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Windows common locations
    win_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
        os.path.expanduser(r"~\Downloads\ffmpeg\bin\ffmpeg.exe"),
    ]
    for p in win_paths:
        if os.path.isfile(p):
            return p

    return None  # FFmpeg nahi mila — skip preprocessing

FFMPEG_PATH = find_ffmpeg()
if FFMPEG_PATH:
    print(f"✅ FFmpeg found: {FFMPEG_PATH}")
else:
    print("⚠️  FFmpeg not found — audio preprocessing disabled (install ffmpeg and add to PATH)")


def preprocess_audio(input_path: str) -> str:
    """
    FFmpeg se mandi audio clean karo:
      - highpass=f=80   → pankhe/fan ki bass remove
      - lowpass=f=8000  → high freq hiss remove
      - loudnorm        → volume normalize
      - 16kHz mono WAV  → Whisper ka optimal format

    Agar FFmpeg nahi mila → original file return karo (no crash)
    """
    if not FFMPEG_PATH:
        return input_path  # FFmpeg nahi — silently skip

    base   = os.path.splitext(input_path)[0]
    out    = base + "_clean.wav"

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-af",
        "highpass=f=100,lowpass=f=7500,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out,
        "-loglevel", "error"
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            print(f"✅ FFmpeg cleaned: {os.path.getsize(out)//1024}KB")
            return out
        print(f"⚠️  FFmpeg failed rc={r.returncode}")
    except subprocess.TimeoutExpired:
        print("⚠️  FFmpeg timeout — using original")
    except Exception as e:
        print(f"⚠️  FFmpeg error: {e}")

    return input_path


# ================================================================
# WHISPER PROMPT — Mandi vocabulary hint
# Yahan jitne zyada mandi words, utna accurate transcription
# ================================================================
WHISPER_PROMPT = (
    "پاکستانی منڈی میں سبزیوں کی خریداری کا بیان: "
    "آلو پیاز ٹماٹر بینگن کریلا گھیا توری مٹر لہسن ادرک مرچ دھنیا پالک میتھی گوبھی شلجم مولی گاجر کدو بھنڈی "
    "من کلو دھڑی پاؤ سیر کوئنٹل "
    "سوا اڑھائی ڈیڑھ پونے آدھا پون ساڑھے "
    "علی زوار احمد اکبر شاہد رشید بشیر نصیر حامد "
    "پانچ سو ہزار پندرہ سو دو ہزار تین ہزار "
    "روپے فی کلو فی من بھاؤ قیمت"
)


# ================================================================
# PHONETIC CORRECTIONS — Whisper ki aaam galtiyan
# ================================================================
PHONETIC_CORRECTIONS = {
    # Wazan
    r'\bمنہ\b': 'من',       r'\bمنوں\b': 'من',
    r'\bکلہ\b': 'کلو',      r'\bکلو گرام\b': 'کلو',
    r'\bدھڑیاں\b': 'دھڑی',
    r'\bپاو\b': 'پاؤ',      r'\bپاوا\b': 'پاؤ',
    r'\bسیرا\b': 'سیر',
    # Fractions
    r'\bدھائی\b': 'اڑھائی', r'\bڈیڑ\b': 'ڈیڑھ',
    r'\bادھا\b': 'آدھا',    r'\bاردھا\b': 'آدھا',
    # Names
    r'\bالی\b': 'علی',       r'\bعلے\b': 'علی',
    r'\bزوارہ\b': 'زوار',   r'\bزوارا\b': 'زوار',
    # Sabziyaan
    r'\bٹماٹرز\b': 'ٹماٹر',
    r'\bپیاذ\b': 'پیاز',    r'\bپیازا\b': 'پیاز',
    r'\bگوبی\b': 'گوبھی',
}


def apply_phonetic_corrections(text: str) -> str:
    for pattern, replacement in PHONETIC_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


# ================================================================
# LLM SYSTEM PROMPT — v3.1 Strongest version
#
# Key improvements:
#  - Roman Urdu → Urdu script conversion explicitly mentioned
#  - More fraction examples
#  - Strict JSON instruction at top AND bottom
#  - No response_format dependency (manual JSON parse)
# ================================================================
MANDI_SYSTEM_PROMPT = """You are a Pakistani mandi (wholesale vegetable market) billing expert. Your ONLY job is to convert speech transcription into clean Urdu billing text.

CRITICAL: You must ALWAYS respond with ONLY this JSON format, nothing else:
{"corrected_text": "your answer here"}

═══════════════════════════════════════
RULE 1 — NUMBERS: Always use digits, NEVER Urdu/Roman words
  ❌ پچاس     →  ✅ 50
  ❌ pachas   →  ✅ 50
  ❌ دو سو    →  ✅ 200
  ❌ do so    →  ✅ 200
  ❌ ہزار     →  ✅ 1000
  ❌ paanch so →  ✅ 500

RULE 2 — FRACTIONS: Convert to exact decimal digits
  سوا / sawa        = base + 0.25   →  سوا من = 1.25 من
  پاؤ / paon        = 0.25          →  پاؤ کلو = 0.25 کلو
  آدھا / adha       = 0.5           →  آدھا من = 0.5 من
  پون / پونے / pone = base - 0.25  →  پون من = 0.75 من  |  پونے دو = 1.75
  ڈیڑھ / dairh      = 1.5           →  ڈیڑھ کلو = 1.5 کلو
  اڑھائی / adhai    = 2.5           →  اڑھائی من = 2.5 من
  ساڑھے X / sarhay  = X + 0.5      →  ساڑھے تین = 3.5

RULE 3 — ROMAN URDU → URDU SCRIPT (very important!)
  aloo        →  آلو          pyaz       →  پیاز
  tamatar     →  ٹماٹر        lehsan     →  لہسن
  adrak       →  ادرک         mirch      →  مرچ
  gobhi       →  گوبھی        palak      →  پالک
  man         →  من           kilo/kg    →  کلو
  dhadi       →  دھڑی         seer       →  سیر
  Ali         →  علی          Zuwar      →  زوار
  Ahmad       →  احمد         Akbar      →  اکبر
  Shahid      →  شاہد         Rashid     →  رشید
  Bashir      →  بشیر         Naseer     →  نصیر
  rupay       →  روپے         fi kilo    →  فی کلو

RULE 4 — KEEP these words in Urdu script:
  Weights: کلو، من، دھڑی، سیر، پاؤ، کوئنٹل
  Vegetables: all veggie names in Urdu
  Names: all person names in Urdu script

═══════════════════════════════════════
EXAMPLES (study these carefully):
  Input:  "ali sawa man aloo pachas rupay"
  Output: {"corrected_text": "علی 1.25 من آلو 50 روپے"}

  Input:  "zuwar adhai man pyaz tees rupay fi kilo"
  Output: {"corrected_text": "زوار 2.5 من پیاز 30 روپے فی کلو"}

  Input:  "احمد ڈیڑھ کلو لہسن پانچ سو"
  Output: {"corrected_text": "احمد 1.5 کلو لہسن 500"}

  Input:  "sarhay teen man gobhi akbar"
  Output: {"corrected_text": "3.5 من گوبھی اکبر"}

  Input:  "pona man tamatar basheer do so"
  Output: {"corrected_text": "0.75 من ٹماٹر بشیر 200"}

  Input:  "ponay do man pyaz rasheed teen so pachaas"
  Output: {"corrected_text": "1.75 من پیاز رشید 350"}

  Input:  "teen dhadi gobhi nau so"
  Output: {"corrected_text": "3 دھڑی گوبھی 900"}

  Input:  "paanch man aloo ek hazaar"
  Output: {"corrected_text": "5 من آلو 1000"}

═══════════════════════════════════════
RESPOND ONLY WITH JSON: {"corrected_text": "..."}
No explanation. No extra text. Just the JSON."""


OR_HEADERS = {
    "HTTP-Referer": "https://mandee.ai",
    "X-Title": "Mandee AI"
}


def extract_json_safe(raw: str, fallback: str) -> str:
    """
    LLM response se JSON extract karo — safely.
    Kuch models extra text ya markdown likhte hain JSON ke saath.
    """
    if not raw:
        return fallback

    # Try direct parse
    try:
        return json.loads(raw).get("corrected_text", fallback)
    except Exception:
        pass

    # JSON block dhundo andar text mein
    match = re.search(r'\{[^{}]*"corrected_text"\s*:\s*"([^"]+)"[^{}]*\}', raw, re.DOTALL)
    if match:
        return match.group(1)

    # Sirf text extract karo agar JSON nahi mila
    # Markdown fence hatao
    cleaned = re.sub(r'```[a-z]*\n?', '', raw).strip()
    try:
        return json.loads(cleaned).get("corrected_text", fallback)
    except Exception:
        pass

    return fallback


def call_llm(model: str, text: str) -> str:
    """
    LLM call — response_format sirf Claude ke saath use karo.
    Gemini models ke liye manual JSON parse.
    """
    use_json_format = "anthropic/" in model  # Only Claude supports it reliably

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": MANDI_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Convert this mandi speech: {text}"}
        ],
        temperature=0,
        max_tokens=200,
        extra_headers=OR_HEADERS,
    )

    if use_json_format:
        kwargs["response_format"] = {"type": "json_object"}

    response = openrouter_client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content.strip()
    return extract_json_safe(raw, text)


def process_text_with_llm(raw_urdu: str) -> dict:
    """
    LLM cascade — speed ke liye Flash first, accuracy ke liye Opus fallback.

    Order changed in v3.1:
      TIER 1: gemini-2.5-flash        ← Fastest (< 1s usually)
      TIER 2: claude-opus-4.6         ← Best accuracy
      TIER 3: gemini-3.1-pro-preview  ← Last resort
    """
    cleaned = apply_phonetic_corrections(raw_urdu)

    tiers = [
        ("google/gemini-2.5-flash",        "TIER1 Flash (fast)"),
        ("anthropic/claude-opus-4.6",      "TIER2 Opus (accurate)"),
        ("google/gemini-3.1-pro-preview",  "TIER3 Gemini Pro"),
    ]

    for model_id, label in tiers:
        try:
            result = call_llm(model_id, cleaned)
            print(f"✅ {label} succeeded")
            return {
                "corrected_text":  result,
                "model_used":      model_id,
                "phonetic_cleaned": cleaned
            }
        except Exception as e:
            print(f"⚠️  {label} failed: {e}")

    print("❌ All LLMs failed")
    return {
        "corrected_text":  cleaned,
        "model_used":      "phonetic_only",
        "phonetic_cleaned": cleaned
    }


# ================================================================
# ENDPOINTS
# ================================================================

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    start_time = time.time()
    ts       = int(time.time() * 1000)
    ext      = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"

    # Windows safe temp path
    tmp_dir  = os.environ.get("TEMP", "/tmp")
    raw_path = os.path.join(tmp_dir, f"mandee_raw_{ts}{ext}")
    clean_path = None

    try:
        # 1. Save raw audio
        audio_bytes = await file.read()
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        file_size = os.path.getsize(raw_path)
        if file_size < 500:
            return {"status": "error", "message": "Audio too small — kuch bola nahi"}

        # 2. FFmpeg clean
        preprocessed = preprocess_audio(raw_path)
        if preprocessed != raw_path:
            clean_path = preprocessed

        # 3. Whisper
        with open(preprocessed, "rb") as af:
            transcription = whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                language="ur",
                prompt=WHISPER_PROMPT,
                response_format="text",
                temperature=0.0
            )

        urdu_raw = (
            transcription.strip()
            if isinstance(transcription, str)
            else transcription.text.strip()
        )

        if not urdu_raw:
            return {"status": "error", "message": "آواز سنائی نہیں دی"}

        # 4. LLM cascade
        llm_result = process_text_with_llm(urdu_raw)

        return {
            "status":           "success",
            "processed_text":   llm_result["corrected_text"],
            "original_whisper": urdu_raw,
            "model_used":       llm_result["model_used"],
            "ffmpeg_active":    FFMPEG_PATH is not None,
            "latency":          f"{time.time() - start_time:.2f}s"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        for path in [raw_path, clean_path]:
            if path and os.path.exists(path):
                try: os.remove(path)
                except: pass


@app.post("/process-text")
async def process_text(data: dict):
    raw = data.get("text", "").strip()
    if not raw:
        return {"status": "error", "message": "متن خالی ہے"}
    t = time.time()
    r = process_text_with_llm(raw)
    return {
        "status":         "success",
        "processed_text": r["corrected_text"],
        "original_text":  raw,
        "model_used":     r["model_used"],
        "latency":        f"{time.time() - t:.2f}s"
    }



@app.get("/health")
async def health():
    return {
        "status":    "online ✅",
        "ffmpeg":    FFMPEG_PATH or "NOT INSTALLED",
        "llm_tier1": "google/gemini-2.5-flash",
        "llm_tier2": "anthropic/claude-opus-4.6",
        "llm_tier3": "google/gemini-3.1-pro-preview",
    }


@app.get("/")
async def root():
    return {"app": "Mandee AI v3.1", "endpoints": ["/process-audio", "/process-text", "/health"]}



