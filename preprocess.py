import pandas as pd
import re, html, unidecode
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

domain_mapping = {
    "FakeNewsNet_PolitiFact" : "Politics",
    "Constraint-21_Covid": "Healthcare",
    "WELFake": "General News",
    "Fakeddit": "General News",
    "LIAR2" : "Politics",
    "Fake News Detection" : "Politics",
    "LLMFake": "AI Generated",
    "FakeNewsNet_GossipCop": "Entertainment",
    "BuzzFeed_Political_News": "Politics",
    "COVID_Lies": "Heathcare",
    "Climate_Fever": "Climate Change",
    "FineFake": "General News"
}

CLEAN_RE = re.compile(r"(https?:\/\/\S+)|[^a-zA-Z\s]", re.MULTILINE)

MODEL = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


df = pd.DataFrame({
    "text": [
        "<p>COVID-19 vaccines ARE safe!</p>",
        "BREAKING: Scientists discover 🍕 on Mars!!!",
        "Get rich quick!! Visit http://scammy.biz NOW!",
        "El niño está aquí – ¡prepárate! ☀️",
        "   Multiple     spaces   and \n newlines.\t\tWow.   "
    ]
})

df.Name = "LIAR2"

def add_domain(dataset:pd.DataFrame) -> pd.DataFrame:
    dataset["domain"] = domain_mapping[dataset.Name]
    return dataset


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    text = unidecode.unidecode(text)

    text = text.lower()

    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str):
    return tokenizer(text["text"], truncation=True)

