import os
import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]          # <repo>/src/backend
PROJECT_DIR = BACKEND_DIR.parents[1]                       # <repo>: holds config/, data/, sample-data/


def _default_data_dir() -> Path:
    default = PROJECT_DIR / "data"
    if os.environ.get("VERCEL") or not os.access(PROJECT_DIR, os.W_OK):
        return Path(tempfile.gettempdir()) / "vitalis-data"
    return default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HVA_", env_file=PROJECT_DIR / "config" / ".env", extra="ignore")

    # Hugging Face Inference Router (OpenAI-compatible API, replaces local/Raspberry Pi Ollama)
    hf_token: str = Field("", validation_alias=AliasChoices("HF_TOKEN", "HVA_HF_TOKEN"))
    hf_base_url: str = "https://router.huggingface.co/v1"
    llm_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    llm_timeout_s: float = 45.0
    llm_num_ctx: int = 4096

    # Patient consult model (served via Hugging Face Inference Router or dedicated HF Inference Endpoint)
    med_model: str = "google/gemma-4-31B-it"
    med_api_url: str = ""
    med_timeout_s: float = 45.0
    med_max_tokens: int = 220

    # Patient consult: RAG over real patient-doctor conversations (Hugging Face datasets), optional local Chroma index
    dialogue_collection: str = "patient_doctor_dialogues"
    dialogue_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dialogue_min_score: float = 0.66
    dialogue_top_k: int = 4

    # Patient consult: nearby specialists (OpenStreetMap; free, no key). Coordinates are rounded before leaving.
    geo_user_agent: str = "vitalis-ai/1.0 (research prototype)"
    geo_precision_digits: int = 2         # 2 decimals ~ 1.1 km
    doctors_max_results: int = 8

    # Answer routing
    llm_confidence_threshold: float = 0.85
    rag_min_score: float = 0.50
    rag_top_k: int = 4
    strict_llm_first: bool = False
    kb_strong_score: float = 0.60

    # Embeddings / vector DB
    embed_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = Field(default_factory=_default_data_dir)
    chroma_collection: str = "clinical_kb"

    # Speech: Hugging Face Whisper Inference API (or optional local faster-whisper) + browser speech synthesis
    whisper_model: str = "openai/whisper-large-v3-turbo"
    whisper_fast_model: str = ""
    whisper_fast_min_logprob: float = -0.5
    whisper_compute_type: str = "int8"
    piper_voice: Path = PROJECT_DIR / "models/piper/en_US-amy-medium.onnx"

    # Online fallback (free, no key): MedlinePlus, openFDA, Wikipedia
    fallback_timeout_s: float = 4.0
    fallback_enabled: bool = True

    # Patient sign-in with Google (OAuth 2.0) or optional email/name demo sign-in
    google_client_id: str = Field("", validation_alias=AliasChoices("AUTH_GOOGLE_ID", "HVA_GOOGLE_CLIENT_ID"))
    google_client_secret: str = Field("", validation_alias=AliasChoices("AUTH_GOOGLE_SECRET", "HVA_GOOGLE_CLIENT_SECRET"))
    public_url: str = Field("", validation_alias=AliasChoices("HVA_PUBLIC_URL"))
    google_redirect_uri: str = Field("", validation_alias=AliasChoices("HVA_GOOGLE_REDIRECT_URI"))
    auth_disabled: bool = False
    local_auth: bool = False
    session_secret: str = Field("", validation_alias=AliasChoices("HVA_SESSION_SECRET"))

    # Security
    api_key: str = ""
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "patients.sqlite"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def dialogue_dir(self) -> Path:
        return self.data_dir / "chroma_dialogues"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    try:
        s.data_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        s.data_dir = Path(tempfile.gettempdir()) / "vitalis-data"
        s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
