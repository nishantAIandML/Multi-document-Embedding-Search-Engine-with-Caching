import hashlib
import re
from datetime import datetime
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ensure punkt and stopwords are available
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
TOKEN_RE = re.compile(r"\b\w+\b")


def clean_text(text: str) -> str:
    text = text.lower()
    # remove simple HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def now_iso():
    return datetime.now().isoformat() + "Z"