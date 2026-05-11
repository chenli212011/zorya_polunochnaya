"""Translation service backed by deep-translator (Google's free web endpoint).

deep-translator is synchronous and not officially supported by Google — it can
rate-limit or break when Google changes its endpoint. We cache results in-process
to minimize repeat calls.
"""

from __future__ import annotations

from functools import lru_cache

from deep_translator import GoogleTranslator


@lru_cache(maxsize=1)
def supported_languages() -> list[dict]:
    """Return Google Translate's supported languages as [{code, name}, …]."""
    raw = GoogleTranslator().get_supported_languages(as_dict=True)
    # raw: {"english": "en", "french": "fr", ...}
    out = [{"code": code, "name": name.title()} for name, code in raw.items()]
    out.sort(key=lambda x: x["name"])
    return out


@lru_cache(maxsize=2048)
def _translate_cached(text: str, target: str, source: str) -> str:
    return GoogleTranslator(source=source, target=target).translate(text) or ""


def translate(text: str, target: str, source: str = "auto") -> str:
    if not text or not text.strip():
        return ""
    return _translate_cached(text, target, source)
