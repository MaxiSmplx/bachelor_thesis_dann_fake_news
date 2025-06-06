# Dataset names
DATASETS = [
    "Climate-FEVER",
    "Fake News Corpus",
    "Fakeddit",
    "FakeNewsNet",
    "FEVER",
    "FineFake",
    "LIAR2",
    "llm-misinformation",
    "MultiFC",
    "Source based FN",
    "WELFake"
]


# Google Drive IDs for raw dataset files
GOOGLE_DRIVE_IDS_RAW = {
    "Climate-FEVER": "176qfoa8JFeox_tof4bmSPjFjQyrbNBc9",
    "Fake News Corpus": "1jWCJvPtXbVAiPU81AEKdW12Uuw6H_cIO",
    "Fakeddit": "12KOJ5IEGPL7tqmsbl3txYq6yD4btekHW",
    "FakeNewsNet": "1HO7Pv0ZaWKhovF7AdqPYIyezFlmXdTTb",
    "FEVER": "1qi_uRSOjXcfuEbaO_wDRCk5Yw6aLWk1y",
    "FineFake": "1W2ks1xrlBy1wyj4Qa_Shlq3s9p02ktLR",
    "LIAR2": "1vLfuOj1aBDBlfwEgDZvPck7w-TTD0eis",
    "llm-misinformation": "16rJ9dQqGAGyqF8T4aP52ViwnQmn1qUcf",
    "MultiFC": "1nJE9OlpcaWEXkwKH5Mp2UJ_bZWt2OHJC",
    "Source based FN": "1zqS7AyzHVA8XrZIkKoNeJlpSKSD026MP",
    "WELFake": "1FjwRjhqZyQLorAtO_k03G13efwMlmsmP"
}

# Google Drive IDs for slighty preprocessed datasets
GOOGLE_DRIVE_IDS = {
    "Climate-FEVER": {
        "real": "1_b0RwUh3YVwT10leIkdIk9XKl3QAa8eg",
        "fake": "1ECNEewI9OlkWM-dv5hU1AERFqo6RGlra"
    },
    "Fake News Corpus": {
        "real": "10KCEx0JEjvn42oej2ptJgH-iAUitdP4k",
        "fake": "143JrvutfvpKQvtDzHlVHXpAeKW9KcpXA"
    },
    "Fakeddit": {
        "real": "1maPEbPoW9SY9h28t5zE3K-8qSsJAldl7",
        "fake": "1o6j4D2V9oTaLcspxIx-O8_2Tz1J3FIUV"
    },
    "FakeNewsNet": {
        "real": "1A9WFJIfDVnfwHXjUXyadbET97tzKnMg_",
        "fake": "1GIoopNM0GlHbA14G55KdOBg7EVD97-c3"
    },
    "FEVER": {
        "real": "1r3A40-CcfOc3_yG0NmDn0mu92FwraUU9",
        "fake": "159pXTV_FIqL3AIGDHV9hAWMtpxceOAz1"
    },
    "FineFake": {
        "real": "1UftWuMbZWglHEeiWZqxTzfYppEEwWwoK",
        "fake": "1miAd7kYRKkGNu8ygDpmNwn5gtl-4hyGy"
    },
    "LIAR2": {
        "real": "15MlZ7_xprEFmbhiRYKWNPkE33OCbwBaj",
        "fake": "1lbXgL_kyFxrx3iYZnbAYjnDDRFJ3g7Zq"
    },
    "llm-misinformation": {
        "real": "1KJxwCKjBMABE8I3NtnZBB-9fyopdWVLv",
        "fake": "1MpWm4ynGnmKRaxabia3Vfu56INfOkID2"
    },
    "MultiFC": {
        "real": "1spVOSpfin3ZTXHh5lbGf9xlWD4EV-yDi",
        "fake": "1pEvHfxsvt3-c3fgJIqc5GL_e9X5jGMHI"
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


# Google Drive IDs for final processed and augmented dataset
GOOGLE_DRIVE_FINAL_IDS = {
    "raw": {
        "train_val": "",
        "test": "",
    }, 
    "balanced": {
        "train_val": "",
        "test": "",
    },
    "augmented": {
        "train_val": "",
        "test": "",
    },
    "balanced_augmented": {
        "train_val": "",
        "test": "",
    }, 
}