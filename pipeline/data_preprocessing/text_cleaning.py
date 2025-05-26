import re
import string
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def remove_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

def remove_urls(text: str) -> str:
     return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

def remove_digits(text: str) -> str:
    return re.sub(r"\d+", "", text)

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_special_characters(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def to_lowercase(text: str) -> str:
    return text.lower()