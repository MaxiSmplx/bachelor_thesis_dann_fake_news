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


GOOGLE_DRIVE_IDS = {
    "Climate-FEVER": {
        "real": "1_b0RwUh3YVwT10leIkdIk9XKl3QAa8eg",
        "fake": "1ECNEewI9OlkWM-dv5hU1AERFqo6RGlra"
    },
    "Fake News Corpus": {
        "real": "10KCEx0JEjvn42oej2ptJgH-iAUitdP4k",
        "fake": "143JrvutfvpKQvtDzHlVHXpAeKW9KcpXA"
    },
    "Fake News Prediction": {
        "real": "1a_nGHNUh7d7rpgdQYatoidHbb0EMHPId",
        "fake": "1wmm9cokUDFzEGiQb2edUxcTubYrfhHFg"
    },
    "Fakeddit": {
        "real": "1maPEbPoW9SY9h28t5zE3K-8qSsJAldl7",
        "fake": "1o6j4D2V9oTaLcspxIx-O8_2Tz1J3FIUV"
    },
    "FEVER": {
        "real": "1r3A40-CcfOc3_yG0NmDn0mu92FwraUU9",
        "fake": "159pXTV_FIqL3AIGDHV9hAWMtpxceOAz1"
    },
    "FineFake": {
        "real": "1UftWuMbZWglHEeiWZqxTzfYppEEwWwoK",
        "fake": "1miAd7kYRKkGNu8ygDpmNwn5gtl-4hyGy"
    },
    "ISOT Fake News": {
        "real": "1TP37DVPIebWZURq9cJYGJBmA0oifRw2a",
        "fake": "1v2g1eF1twUtFAZmA2dlVpf1pn4hLZz0o"
    },
    "LIAR2": {
        "real": "15MlZ7_xprEFmbhiRYKWNPkE33OCbwBaj",
        "fake": "1lbXgL_kyFxrx3iYZnbAYjnDDRFJ3g7Zq"
    },
    "llm-misinformation": {
        "real": "1KJxwCKjBMABE8I3NtnZBB-9fyopdWVLv",
        "fake": "1MpWm4ynGnmKRaxabia3Vfu56INfOkID2"
    },
    "Source based FN": {
        "real": "1Ij7Dt6AcMXfSEykuMHQtcCsfw_mE9-gn",
        "fake": "1WszJIzt5PxAYZAaZ3B4sYlcCxNk2J_Rj"
    },
    "WELFake": {
        "real": "1pFzGonPbzo8CCGSNTGtjDOTqI5l3hARg",
        "fake": "162x2UNvEkQ56eNf6ws05gFzPzHfsDPfm"
    }
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