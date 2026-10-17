from pathlib import Path

# Repository root (config.py lives in src/). Data, cache and outputs are resolved relative to it.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ml-1m"
CACHE_DIR = ROOT / ".cache"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET = "ml-1m"
TEST_RATIO = 0.2
HISTORY_MAX = 20

import os as _os
LLM_BASE_URL = _os.environ.get("CF_LLM_BASE_URL", "http://localhost:8002/v1")
LLM_MODEL = _os.environ.get("CF_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
LLM_API_KEY = _os.environ.get("CF_LLM_API_KEY", "dummy")  # vLLM ignores it unless --api-key is set
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 3
LLM_CONCURRENCY = 16

K_EXPLANATIONS = 5
TIPS_WINDOW_DAYS = 30
TIPS_CLIP_MAX = 10.0
TIPS_ALPHA = 0.5

SUBSAMPLE_N = 500
POPULARITY_BUCKETS = 5

AGE_GROUPS = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}

OCCUPATIONS = {
    0: "other or not specified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
    13: "retired", 14: "sales/marketing", 15: "scientist",
    16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
    19: "unemployed", 20: "writer",
}
