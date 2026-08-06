import io

def generate_audio_bytes(text):
    """
    Generates MP3 audio bytes for the given text using gTTS (Google Text-to-Speech).
    Returns audio bytes or None on failure.
    """
    if not text or not text.strip():
        return None

    try:
        from gtts import gTTS
        # Truncate text to 3,000 chars max to keep audio generation fast
        sample_text = text[:3000].strip()
        tts = gTTS(text=sample_text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"Text-to-Speech error: {e}")
        return None
