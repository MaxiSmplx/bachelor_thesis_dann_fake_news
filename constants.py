DATASETS = [
    "Climate-FEVER",
    "Fake News Corpus",
    "Fake News Prediction",
    "Fakeddit",
    "FEVER",
    "FineFake",
    "ISOT Fake News",
    "LIAR2",
    "llm-misinformation",
    "Source based FN",
    "WELFake"
]


GOOGLE_DRIVE_IDS_RAW = {
    "Climate-FEVER": "176qfoa8JFeox_tof4bmSPjFjQyrbNBc9",
    "Fake News Corpus": "1jWCJvPtXbVAiPU81AEKdW12Uuw6H_cIO",
    "Fake News Prediction": "1R9mnhhPQE62-pTdPaY29MyOpP7B0ZxOK",
    "Fakeddit": "12KOJ5IEGPL7tqmsbl3txYq6yD4btekHW",
    "FEVER": "1qi_uRSOjXcfuEbaO_wDRCk5Yw6aLWk1y",
    "FineFake": "1W2ks1xrlBy1wyj4Qa_Shlq3s9p02ktLR",
    "ISOT Fake News": "14BZ3Fi0S7M362fxGQHZ4mYGGlwcnr9wf",
    "LIAR2": "1vLfuOj1aBDBlfwEgDZvPck7w-TTD0eis",
    "llm-misinformation": "16rJ9dQqGAGyqF8T4aP52ViwnQmn1qUcf",
    "Source based FN": "1zqS7AyzHVA8XrZIkKoNeJlpSKSD026MP",
    "WELFake": "1FjwRjhqZyQLorAtO_k03G13efwMlmsmP"
}


METADATA = {
    "WELFake": {
        "name": "WELFake",
        "columns" : ["title","text", "label"],
        "domain": "",
    },
    "Source based FN": {
        "name": "Source based FN",
        "columns" : ["title","text", "label"],
        "domain": "",
    },
    "llm-misinformation": {
        "name": "llm-misinformation",
        "columns" : ["label", "synthetic_misinformation"],
        "domain": "",
    },
    "LIAR2": {
        "name": "LIAR2",
        "columns" : ["id", "label", "statement", "speaker"],
        "domain": "",
    },
    "ISOT Fake News": {
        "name": "ISOT Fake News",
        "columns" : ["title", "text", "label"],
        "domain": "",
        },
    "FineFake": {
        "name": "FineFake",
        "columns" : ["text", "label"],
        "domain": "",
    },
    "FEVER": {
        "name": "FEVER",
        "columns" : ["label", "claim"],
        "domain": "",
    },
    "Fakeddit": {
        "name": "Fakeddit",
        "columns" : ["title", "label"],
        "domain": "",
    },
    "Fake News Prediction": {
        "name": "Fake News Prediction",
        "columns" : ["title", "text", "label"],
        "domain": "",
    },
    "Fake News Corpus": {
        "name": "Fake News Corpus",
        "columns" : ["content", "title", "label"],
        "domain": "",
    },
    "Climate-FEVER": {
        "name": "Climate-FEVER",
        "columns" : ["claim", "label"],
        "domain": "",
    }
}