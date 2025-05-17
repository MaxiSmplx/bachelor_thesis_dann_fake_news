import random
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.data import find
from nltk import download

def ensure_nltk_data():
    # Map each NLTK package (except wordnet) to its expected resource paths
    _RESOURCE_PATHS = {
        'punkt': ['tokenizers/punkt'],
        'punkt_tab': ['tokenizers/punkt_tab/english'],
    }
    # The Perceptron tagger always looks for the “_eng” folder at runtime
    _TAGGER_ENG_PATH = 'taggers/averaged_perceptron_tagger_eng'

    # What to pass to nltk.download()
    _DOWNLOAD_NAMES = {
        'punkt': ['punkt'],
        'punkt_tab': ['punkt_tab'],
        'averaged_perceptron_tagger': ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'],
    }

    # 1) punkt & punkt_tab
    for pkg, paths in _RESOURCE_PATHS.items():
        missing = True
        for path in paths:
            try:
                find(path)
                missing = False
                break
            except LookupError:
                continue
        if missing:
            print(f"NLTK data '{pkg}' not found; downloading…")
            for name in _DOWNLOAD_NAMES[pkg]:
                download(name, quiet=True)

    # 2) averaged_perceptron_tagger → check only the _eng JSON folder
    try:
        find(_TAGGER_ENG_PATH)
    except LookupError:
        print("NLTK data 'averaged_perceptron_tagger_eng' not found; downloading both tagger packages…")
        for name in _DOWNLOAD_NAMES['averaged_perceptron_tagger']:
            download(name, quiet=True)

    # 3) wordnet → test via a simple lookup
    try:
        # if WordNet data is present, this will return a (possibly empty) list
        _ = wordnet.synsets('test')
    except LookupError:
        print("NLTK data 'wordnet' not found; downloading…")
        download('wordnet', quiet=True)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None


def get_synonyms(word, pos=None):
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    synonyms = {
        lemma.name().replace('_', ' ')
        for syn in synsets
        for lemma in syn.lemmas()
        if lemma.name().lower() != word.lower()
    }
    return list(synonyms)


def replace_synonyms_database(sentence:str, n_replacements:int = 1) -> str:
    ensure_nltk_data()

    words = word_tokenize(sentence)
    tagged = pos_tag(words)

    candidates = [
        (i, w, get_wordnet_pos(tag))
        for i, (w, tag) in enumerate(tagged)
        if w.isalpha()
    ]
    random.shuffle(candidates)

    new_words = words[:]
    replaced = 0
    for idx, word, wn_pos in candidates:
        if replaced >= n_replacements:
            break
        syns = get_synonyms(word, pos=wn_pos)
        if not syns:
            continue
        new_words[idx] = random.choice(syns)
        replaced += 1

    return ' '.join(new_words)



#--------------------------------


from data_augmentation._openai_api import call_openai

def replace_synonyms_llm(text: str, synonym_replacements_n: int) -> str:
    context = f"You are a lexical refinement expert: a concise, context-aware assistant skilled at swapping out words for single-word synonyms that perfectly match the original texts tone, style, and meaning."
    prompt = f"In the text below, replace exactly {synonym_replacements_n} words with single-word synonyms that fit the context and preserve the original tone and semantics. Return only the revised text, with no explanations or formatting: {text}"

    return call_openai(context, prompt)
