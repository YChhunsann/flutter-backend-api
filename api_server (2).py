"""
FastAPI Server for VITS Text-to-Speech
Provides endpoints for Flutter frontend to generate speech from text
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import json
import io
import base64
import numpy as np
import torch
from scipy.io.wavfile import write
from typing import Optional
import csv

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

app = FastAPI(
    title="VITS TTS API",
    description="Text-to-Speech API for VITS model (Khmer)",
    version="1.0.0"
)

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and config
net_g = None
hps = None
device = None
lexicon = None
lexicon_path = None


# Request/Response models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Input text (Khmer or phoneme)", min_length=1)
    noise_scale: Optional[float] = Field(0.667, description="Controls speech variation", ge=0.0, le=2.0)
    noise_scale_w: Optional[float] = Field(0.8, description="Controls prosody variation", ge=0.0, le=2.0)
    length_scale: Optional[float] = Field(1.0, description="Controls speech speed", ge=0.1, le=3.0)
    is_phoneme: Optional[bool] = Field(False, description="Set true if `text` is already phoneme")


class TTSBase64Response(BaseModel):
    success: bool
    audio: str
    format: str
    sample_rate: int
    text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model: str
    language: str
    sampling_rate: int
    symbols_count: int
    text_cleaners: list
    device: str
    lexicon_loaded: bool
    lexicon_path: Optional[str]


########################################
# Khmer Lexicon-Based Phonemizer (embedded)
########################################

_punctuation = ';:!?—…«»“” '
_full_stop = '។៕'
_boundary = set(_punctuation + _full_stop)


def load_lexicon(lex_path):
    """Load TSV lexicon: word<TAB>phoneme. Also auto-split compound entries.
    Mirrors logic from text-proprocessing/text_to_phoneme.py.
    """
    lx = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            word, phonemes = line.split("\t", 1)
            word = word.strip()
            phonemes = phonemes.strip()
            lx[word] = phonemes

            phone_units = [p.strip() for p in phonemes.split(".")]
            if len(phone_units) > 1:
                chars = list(word)
                avg = max(1, len(chars) // len(phone_units))
                idx = 0
                for ph in phone_units:
                    sub = "".join(chars[idx:idx + avg])
                    if sub and sub not in lx:
                        lx[sub] = ph
                    idx += avg
    return lx


def tokenize_text(text, lx):
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _boundary:
            tokens.append(ch)
            i += 1
            continue
        match = None
        for j in range(len(text), i, -1):
            chunk = text[i:j]
            if chunk in lx:
                match = chunk
                break
        if match:
            tokens.append(match)
            i += len(match)
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def tokens_to_phoneme(tokens, lx):
    output = []
    for tok in tokens:
        if tok in _boundary:
            output.append(tok)
        elif tok in lx:
            output.append(lx[tok])
        else:
            output.append("UNK")
    return ".".join(output)


def khmer_text_to_phoneme(text, lx):
    tokens = tokenize_text(text, lx)
    return tokens_to_phoneme(tokens, lx)


def _find_lexicon_path():
    """Try common locations for lexicon.tsv."""
    candidates = [
        os.path.join(os.getcwd(), "lexicon.tsv"),
        os.path.join(os.getcwd(), "text-proprocessing", "lexicon_new.tsv"),
        # os.path.join(os.getcwd(), "knltk", "lexicon.tsv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def initialize_model():
    """Initialize the VITS model and load checkpoint"""
    global net_g, hps, device, lexicon, lexicon_path
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load lexicon
    lexicon_path = _find_lexicon_path()
    if lexicon_path is None:
        raise RuntimeError("lexicon.tsv not found. Place it in project root or text-proprocessing/ or knltk/.")
    lexicon = load_lexicon(lexicon_path)
    print(f"Loaded lexicon entries: {len(lexicon)} from {lexicon_path}")

    # Load hyperparameters
    config_path = "./logs/khmer_vits_model/config.json"
    hps = utils.get_hparams_from_file(config_path)
    
    # Initialize model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = "./logs/khmer_vits_model/G_79000.pth"
    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    net_g.eval()
    
    print("Model initialized successfully!")


def get_text(text, hps):
    """Convert text to tensor sequence"""
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def generate_audio(text, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
    """Generate audio from text using VITS model"""
    global net_g, hps, device
    
    if net_g is None:
        raise Exception("Model not initialized")
    
    # Preprocess text
    stn_tst = get_text(text, hps)
    
    # Generate audio
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(
            x_tst, 
            x_tst_lengths, 
            noise_scale=noise_scale, 
            noise_scale_w=noise_scale_w, 
            length_scale=length_scale
        )[0][0, 0].data.cpu().float().numpy()
    
    return audio


@app.get('/health', response_model=HealthResponse, tags=["Status"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status='healthy',
        model_loaded=net_g is not None
    )

@app.get('/')
async def root():
    """Root endpoint"""
    return {"message": "VITS TTS API is running", "docs": "/docs"}

@app.post('/tts', tags=["Text-to-Speech"])
async def text_to_speech(request: TTSRequest):
    """
    Main TTS endpoint - Returns WAV audio file
    
    - **text**: Phoneme text to synthesize (required)
    - **noise_scale**: Controls speech variation (0.0-2.0, default: 0.667)
    - **noise_scale_w**: Controls prosody variation (0.0-2.0, default: 0.8)
    - **length_scale**: Controls speech speed (0.1-3.0, default: 1.0)
    """
    try:
        if net_g is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # If input is Khmer text, convert to phoneme first
        input_text = request.text
        if not request.is_phoneme:
            if lexicon is None:
                raise HTTPException(status_code=500, detail="Lexicon not loaded")
            input_text = khmer_text_to_phoneme(input_text, lexicon)

        # Generate audio
        audio = generate_audio(
            input_text, 
            request.noise_scale, 
            request.noise_scale_w, 
            request.length_scale
        )
        
        # Convert to WAV format in memory
        audio_clip = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clip * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_io = io.BytesIO()
        write(wav_io, hps.data.sampling_rate, audio_int16)
        wav_io.seek(0)
        
        # Return audio file
        return StreamingResponse(
            wav_io,
            media_type='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename="synthesized_speech.wav"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/tts/base64', response_model=TTSBase64Response, tags=["Text-to-Speech"])
async def text_to_speech_base64(request: TTSRequest):
    """
    TTS endpoint that returns base64 encoded audio in JSON
    
    - **text**: Phoneme text to synthesize (required)
    - **noise_scale**: Controls speech variation (0.0-2.0, default: 0.667)
    - **noise_scale_w**: Controls prosody variation (0.0-2.0, default: 0.8)
    - **length_scale**: Controls speech speed (0.1-3.0, default: 1.0)
    """
    try:
        if net_g is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # If input is Khmer text, convert to phoneme first
        input_text = request.text
        if not request.is_phoneme:
            if lexicon is None:
                raise HTTPException(status_code=500, detail="Lexicon not loaded")
            input_text = khmer_text_to_phoneme(input_text, lexicon)

        # Generate audio
        audio = generate_audio(
            input_text, 
            request.noise_scale, 
            request.noise_scale_w, 
            request.length_scale
        )
        
        # Convert to WAV format in memory
        audio_clip = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clip * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_io = io.BytesIO()
        write(wav_io, hps.data.sampling_rate, audio_int16)
        wav_bytes = wav_io.getvalue()
        
        # Encode to base64
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Return JSON response
        return TTSBase64Response(
            success=True,
            audio=audio_base64,
            format='wav',
            sample_rate=int(hps.data.sampling_rate),
            text=request.text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in TTS base64 endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/info', response_model=InfoResponse, tags=["Status"])
async def get_info():
    """Get model information and configuration"""
    if hps is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return InfoResponse(
        model='VITS TTS',
        language='Khmer',
        sampling_rate=hps.data.sampling_rate,
        symbols_count=len(symbols),
        text_cleaners=hps.data.text_cleaners,
        device=str(device),
        lexicon_loaded=lexicon is not None,
        lexicon_path=lexicon_path
    )


@app.post('/phonemize', tags=["Text-to-Speech"])
async def phonemize_text(body: TTSRequest):
    """Utility endpoint for Flutter devs to verify phonemization."""
    try:
        if lexicon is None:
            raise HTTPException(status_code=500, detail="Lexicon not loaded")
        ph = body.text if body.is_phoneme else khmer_text_to_phoneme(body.text, lexicon)
        return {"phoneme": ph, "is_phoneme": body.is_phoneme, "lexicon_path": lexicon_path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    print("Initializing VITS TTS API Server...")
    initialize_model()
    print("API Server ready!")


if __name__ == '__main__':
    import uvicorn
    print("Starting FastAPI server on http://0.0.0.0:5000")
    print("API docs available at http://0.0.0.0:5000/docs")
    uvicorn.run(app, host='127.0.0.1', port=5000)
