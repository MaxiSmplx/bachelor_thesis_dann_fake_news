import pandas as pd
import re, string
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, pipeline
import nlpaug.augmenter.word as naw


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
def add_domain(dataset:pd.DataFrame) -> pd.DataFrame:
    dataset["domain"] = domain_mapping[dataset.Name]
    return dataset


_URL   = re.compile(r"https?://\S+|www\.\S+")
_DIGIT = re.compile(r"\d+")
_HTML  = re.compile(r"<.*?>")
_PUNCT = str.maketrans("", "", string.punctuation)
def clean(text: str) -> str:
    """Lower-case, strip HTML, URLs, digits, punctuation and collapse whitespace."""
    text = BeautifulSoup(text, "lxml").get_text(" ", strip=True)
    text = _URL.sub(" ", text)
    text = _DIGIT.sub(" ", text)
    text = text.translate(_PUNCT)
    text = _HTML.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


MODEL = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
def tokenize(text: str):
    return tokenizer(text["text"], truncation=True)


_syn_aug = naw.SynonymAug(aug_src="wordnet")
def aug_synonyms(text: str, n: int = 2) -> list[str]:
    return [_syn_aug.augment(text) for _ in range(n)]


_para_pipe = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", max_length=256)
def aug_paraphrase(text: str, n: int = 2) -> list[str]:
    return [p["generated_text"] for p in _para_pipe([text] * n)]


_style_pipe = pipeline("text2text-generation", model="s-nlp/t5-informal", max_length=256)
def style_transfer(text: str, target: str = "formal", n: int = 1) -> list[str]:
    prompts = [f"transfer {target}: {text}" for _ in range(n)]
    return [p["generated_text"] for p in _style_pipe(prompts)]



