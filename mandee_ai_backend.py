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

# ================================================================
#  MANDEE AI — v3.2 VERCEL EDITION
#
#  Changes in v3.2:
#    ✅ Vercel environment auto-detect
#    ✅ DEBUG_DIR → /tmp on Vercel (writable)
#    ✅ FFmpeg + preprocessing skipped on Vercel
#    ✅ Debug file writes skipped on Vercel (optional /tmp)
#    ✅ os.path.abspath(__file__) crash fixed for Vercel
#    ✅ uvicorn import removed (not needed on Vercel)
#
#  LLM ORDER (speed + accuracy balanced):
#    TIER 1: google/gemini-2.5-flash        ← Fastest, very good Urdu
#    TIER 2: anthropic/claude-opus-4-5      ← Best accuracy fallback
#    TIER 3: google/gemini-2.5-pro-preview  ← Last resort
# ================================================================

load_dotenv()

# ================================================================
# ENVIRONMENT DETECTION
# ================================================================
IS_VERCEL = os.environ.get("VERCEL") == "1" or "VERCEL_ENV" in os.environ

# ================================================================
# DEBUG AUDIO FOLDER
# Vercel par /tmp use hota hai (read-write), local par debug_audio/
# ================================================================
if IS_VERCEL:
    DEBUG_DIR = "/tmp"
else:
    try:
        _base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _base = os.getcwd()
    DEBUG_DIR = os.path.join(_base, "debug_audio")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"📁 Debug audio folder: {DEBUG_DIR}")


# ================================================================
# CLIENTS
# ================================================================
whisper_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

app = FastAPI(title="Mandee AI v3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# FFMPEG — Windows + Linux/Mac auto-detect
# Vercel par skip kiya jata hai
# ================================================================
def find_ffmpeg() -> str | None:
    if IS_VERCEL:
        return None  # Vercel par FFmpeg available nahi hota

    found = shutil.which("ffmpeg")
    if found:
        return found

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

    return None


FFMPEG_PATH = find_ffmpeg()
if not IS_VERCEL:
    if FFMPEG_PATH:
        print(f"✅ FFmpeg found: {FFMPEG_PATH}")
    else:
        print("⚠️  FFmpeg not found — audio preprocessing disabled (install ffmpeg and add to PATH)")


def preprocess_audio(input_path: str) -> str:
    """
    FFmpeg se audio clean karo.
    Vercel par ya agar FFmpeg nahi mila → original file return karo (no crash).
    """
    if not FFMPEG_PATH or IS_VERCEL:
        return input_path  # Skip on Vercel or when FFmpeg missing

    base = os.path.splitext(input_path)[0]
    out  = base + "_clean.wav"

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-af",
        "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out,
        "-loglevel", "error"
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            print(f"✅ FFmpeg cleaned: {os.path.getsize(out) // 1024}KB")
            return out
        print(f"⚠️  FFmpeg failed rc={r.returncode}")
    except subprocess.TimeoutExpired:
        print("⚠️  FFmpeg timeout — using original")
    except Exception as e:
        print(f"⚠️  FFmpeg error: {e}")

    return input_path


# ================================================================
# WHISPER PROMPT
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
# PHONETIC CORRECTIONS
# ================================================================
PHONETIC_CORRECTIONS = {
    r'\bمنہ\b': 'من',       r'\bمنوں\b': 'من',
    r'\bکلہ\b': 'کلو',      r'\bکلو گرام\b': 'کلو',
    r'\bدھڑیاں\b': 'دھڑی',
    r'\bپاو\b': 'پاؤ',      r'\bپاوا\b': 'پاؤ',
    r'\bسیرا\b': 'سیر',
    r'\bدھائی\b': 'اڑھائی', r'\bڈیڑ\b': 'ڈیڑھ',
    r'\bادھا\b': 'آدھا',    r'\bاردھا\b': 'آدھا',
    r'\bالی\b': 'علی',       r'\bعلے\b': 'علی',
    r'\bزوارہ\b': 'زوار',   r'\bزوارا\b': 'زوار',
    r'\bٹماٹرز\b': 'ٹماٹر',
    r'\bپیاذ\b': 'پیاز',    r'\bپیازا\b': 'پیاز',
    r'\bگوبی\b': 'گوبھی',
}


def apply_phonetic_corrections(text: str) -> str:
    for pattern, replacement in PHONETIC_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


# ================================================================
# LLM SYSTEM PROMPT
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
    if not raw:
        return fallback

    try:
        return json.loads(raw).get("corrected_text", fallback)
    except Exception:
        pass

    match = re.search(r'\{[^{}]*"corrected_text"\s*:\s*"([^"]+)"[^{}]*\}', raw, re.DOTALL)
    if match:
        return match.group(1)

    cleaned = re.sub(r'```[a-z]*\n?', '', raw).strip()
    try:
        return json.loads(cleaned).get("corrected_text", fallback)
    except Exception:
        pass

    return fallback


def call_llm(model: str, text: str) -> str:
    use_json_format = "anthropic/" in model

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
    cleaned = apply_phonetic_corrections(raw_urdu)

    tiers = [
        ("google/gemini-2.5-flash",       "TIER1 Flash (fast)"),
        ("anthropic/claude-opus-4-5",     "TIER2 Opus (accurate)"),
        ("google/gemini-2.5-pro-preview", "TIER3 Gemini Pro"),
    ]

    for model_id, label in tiers:
        try:
            result = call_llm(model_id, cleaned)
            print(f"✅ {label} succeeded")
            return {
                "corrected_text":   result,
                "model_used":       model_id,
                "phonetic_cleaned": cleaned
            }
        except Exception as e:
            print(f"⚠️  {label} failed: {e}")

    print("❌ All LLMs failed")
    return {
        "corrected_text":   cleaned,
        "model_used":       "phonetic_only",
        "phonetic_cleaned": cleaned
    }


# ================================================================
# HELPER: safe debug file write (no crash on Vercel /tmp issues)
# ================================================================
def _safe_write(path: str, content: bytes | str) -> bool:
    try:
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with open(path, mode, **kwargs) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"⚠️  Debug write failed ({path}): {e}")
        return False


# ================================================================
# ENDPOINTS
# ================================================================

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    start_time = time.time()
    ts  = int(time.time() * 1000)
    ext = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"

    # Always use /tmp — safe on both Vercel and local
    tmp_dir    = "/tmp" if IS_VERCEL else os.environ.get("TEMP", "/tmp")
    raw_path   = os.path.join(tmp_dir, f"mandee_raw_{ts}{ext}")
    clean_path = None

    try:
        # 1. Save raw audio
        audio_bytes = await file.read()
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        file_size = os.path.getsize(raw_path)
        if file_size < 500:
            return {"status": "error", "message": "Audio too small — kuch bola nahi"}

        # DEBUG: raw audio save (local only)
        if not IS_VERCEL:
            debug_raw = os.path.join(DEBUG_DIR, f"{ts}_1_RAW{ext}")
            _safe_write(debug_raw, audio_bytes)
            print(f"💾 Saved raw:   {debug_raw}  ({file_size // 1024}KB)")

        # 2. FFmpeg clean (skipped on Vercel)
        preprocessed = preprocess_audio(raw_path)
        if preprocessed != raw_path:
            clean_path = preprocessed
            if not IS_VERCEL:
                debug_clean = os.path.join(DEBUG_DIR, f"{ts}_2_CLEAN.wav")
                shutil.copy2(preprocessed, debug_clean)
                print(f"💾 Saved clean: {debug_clean}")

        # 3. Whisper transcription
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

        # DEBUG: log file (local only — Vercel pe console logs hi kaafi hain)
        if not IS_VERCEL:
            log_path = os.path.join(DEBUG_DIR, f"{ts}_3_LOG.txt")
            log_content = (
                f"=== MANDEE AI DEBUG LOG ===\n"
                f"Time       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Audio size : {file_size} bytes\n"
                f"FFmpeg     : {'ON' if FFMPEG_PATH else 'OFF'}\n"
                f"\n--- WHISPER OUTPUT (RAW) ---\n{urdu_raw}\n"
                f"\n--- PHONETIC CLEANED ---\n{llm_result['phonetic_cleaned']}\n"
                f"\n--- FINAL LLM OUTPUT ---\n{llm_result['corrected_text']}\n"
                f"\n--- MODEL USED ---\n{llm_result['model_used']}\n"
                f"\nLatency: {time.time() - start_time:.2f}s\n"
            )
            _safe_write(log_path, log_content)
            print(f"💾 Saved log:   {log_path}")

        print(f"📝 Whisper said: '{urdu_raw}'")
        print(f"✅ Final output: '{llm_result['corrected_text']}'")

        return {
            "status":           "success",
            "processed_text":   llm_result["corrected_text"],
            "original_whisper": urdu_raw,
            "phonetic_cleaned": llm_result["phonetic_cleaned"],
            "model_used":       llm_result["model_used"],
            "ffmpeg_active":    FFMPEG_PATH is not None,
            "latency":          f"{time.time() - start_time:.2f}s"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        for path in [raw_path, clean_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


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


@app.get("/debug")
async def debug_list():
    """
    Local only: debug_audio folder mein saved files aur logs dikhao.
    Vercel par sirf basic info return hoti hai.
    """
    if IS_VERCEL:
        return {
            "message": "Debug file logging is disabled on Vercel. Check your deployment logs instead.",
            "vercel": True
        }

    if not os.path.exists(DEBUG_DIR):
        return {"debug_dir": DEBUG_DIR, "sessions": []}

    sessions = {}
    for fname in sorted(os.listdir(DEBUG_DIR)):
        ts_part = fname.split("_")[0]
        if ts_part not in sessions:
            sessions[ts_part] = {"files": [], "log": None}
        fpath = os.path.join(DEBUG_DIR, fname)
        sessions[ts_part]["files"].append(fname)
        if fname.endswith("_LOG.txt"):
            try:
                with open(fpath, encoding="utf-8") as f:
                    sessions[ts_part]["log"] = f.read()
            except Exception:
                pass

    result = []
    for ts_key, data in sorted(sessions.items(), reverse=True)[:20]:
        result.append({
            "session_id": ts_key,
            "files": data["files"],
            "log": data["log"]
        })

    return {
        "debug_dir": DEBUG_DIR,
        "total_sessions": len(sessions),
        "last_20_sessions": result
    }


@app.delete("/debug/clear")
async def debug_clear():
    """debug_audio folder khali karo (local only)"""
    if IS_VERCEL:
        return {"message": "Not applicable on Vercel."}

    count = 0
    for fname in os.listdir(DEBUG_DIR):
        try:
            os.remove(os.path.join(DEBUG_DIR, fname))
            count += 1
        except Exception:
            pass
    return {"deleted": count, "message": f"{count} files deleted from {DEBUG_DIR}"}


@app.get("/health")
async def health():
    return {
        "status":        "online ✅",
        "version":       "v3.2 Vercel Edition",
        "environment":   "vercel" if IS_VERCEL else "local",
        "ffmpeg":        FFMPEG_PATH or ("N/A on Vercel" if IS_VERCEL else "NOT INSTALLED (install ffmpeg + add to PATH)"),
        "llm_tier1":     "google/gemini-2.5-flash      — fast primary",
        "llm_tier2":     "anthropic/claude-opus-4-5    — accurate fallback",
        "llm_tier3":     "google/gemini-2.5-pro-preview — last resort",
        "fixes_v3.2": [
            "Vercel environment auto-detect (IS_VERCEL flag)",
            "DEBUG_DIR → /tmp on Vercel",
            "FFmpeg + preprocessing skipped on Vercel",
            "Debug file writes skipped on Vercel (no disk writes)",
            "os.path.abspath(__file__) NameError fixed",
            "uvicorn import removed (not needed on Vercel)",
            "tmp_dir always /tmp on Vercel",
        ]
    }


@app.get("/")
async def root():
    return {
        "app": "Mandee AI v3.2",
        "environment": "vercel" if IS_VERCEL else "local",
        "endpoints": ["/process-audio", "/process-text", "/health", "/debug"]
    }
