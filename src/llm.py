import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import (
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE,
    LLM_MAX_TOKENS, LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_CONCURRENCY,
    CACHE_DIR,
)

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)

_CACHE_PATH = CACHE_DIR / "llm_cache.sqlite"
_cache_lock = threading.Lock()


def _db():
    conn = sqlite3.connect(_CACHE_PATH, timeout=30, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    return conn


_conn = _db()


def _key(model, messages, params):
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(k):
    with _cache_lock:
        row = _conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    return row[0] if row else None


def _cache_put(k, v):
    with _cache_lock:
        _conn.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (k, v))
        _conn.commit()


def chat(messages, *, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
         model=LLM_MODEL, response_format=None, extra_tag=None, use_cache=True):
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "extra_tag": extra_tag,
    }
    k = _key(model, messages, params)
    if use_cache:
        hit = _cache_get(k)
        if hit is not None:
            return hit

    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            kwargs = dict(model=model, messages=messages, temperature=temperature,
                          max_tokens=max_tokens)
            if response_format:
                kwargs["response_format"] = response_format
            resp = _client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            if use_cache:
                _cache_put(k, text)
            return text
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} retries: {last_err}")


def chat_json(messages, *, schema_hint=None, **kwargs):
    text = chat(messages, response_format={"type": "json_object"}, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def chat_logprobs_first_token(messages, *, top_logprobs=10, temperature=0.0,
                              model=LLM_MODEL, max_tokens=4, extra_tag=None,
                              use_cache=True):
    """Ask for a short completion and return a dict {token: prob} from the
    top_logprobs of the FIRST output token only. Used to compute expected
    rating = sum_k k * p(k) over digit tokens {'1','2','3','4','5'}."""
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "extra_tag": extra_tag,
    }
    k = _key(model, messages, params)
    if use_cache:
        hit = _cache_get(k)
        if hit is not None:
            return json.loads(hit)

    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, logprobs=True, top_logprobs=top_logprobs,
            )
            content_lp = resp.choices[0].logprobs
            first = None
            if content_lp and content_lp.content:
                first = content_lp.content[0]
            text = resp.choices[0].message.content or ""
            toks = {}
            if first and first.top_logprobs:
                import math
                for t in first.top_logprobs:
                    toks[t.token] = math.exp(t.logprob)
            out = {"text": text, "first_token": first.token if first else None, "tokens": toks}
            if use_cache:
                _cache_put(k, json.dumps(out))
            return out
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM logprobs call failed after {LLM_MAX_RETRIES} retries: {last_err}")


def batch_map(fn, items, concurrency=LLM_CONCURRENCY, progress=None):
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for f in as_completed(futures):
            i = futures[f]
            results[i] = f.result()
            if progress is not None:
                progress.update(1)
    return results
