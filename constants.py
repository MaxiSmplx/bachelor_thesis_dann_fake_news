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
    "Climate-FEVER": "1eyQDxxa2eRGPlW3eq7ZVCd_0HoU8vEZG",
    "Fake News Corpus": "1N4UnvtM54N48Y1b8TYw7JH8Jx2YbSN9I",
    "Fake News Prediction": "16vMqXRi3KzKvkViI3V0Dp3Okxi2ldHOt",
    "Fakeddit": "16ZW_oaTL3k9Igd20Yt_CyezotKkkb_6I",
    "FEVER": "1ggT5n4G7fI84trKlcW8grzzQUX2Sj582",
    "FineFake": "1Z86KZ00KUfdCr-l3T9w6-pD880wKnaua",
    "ISOT Fake News": "1KcuRj6VIPLFQrW3yNc-mFqhIK24EZumL",
    "LIAR2": "1fO6RQngKVejGUS9xzhSfnMsBlDhx27ot",
    "llm-misinformation": "1wMj6hR9qiDa6_lYelf7N-XmWQnp4hvpD",
    "Source based FN": "1pVPcfZFCturLdLdNEEW7HoL04FdB0dmE",
    "WELFake": "1UHP9yvL76PkO1O80vJFl78rRJpJq4YG0"
}


METADATA = {
    "WELFake": {
        "name": "WELFake",
        "columns" : ["title","text", "label"],
        "domain": "",
        "raw_type" : "csv"
    },
    "Source based FN": {
        "name": "Source based FN",
        "columns" : ["title","text", "label"],
        "domain": "",
        "raw_type" : "csv"
    },
    "llm-misinformation": {
        "name": "llm-misinformation",
        "columns" : ["label", "synthetic_misinformation"],
        "domain": "",
        "raw_type" : "csv"
    },
    "LIAR2": {
        "name": "LIAR2",
        "columns" : ["id", "label", "statement", "speaker"],
        "domain": "",
        "raw_type" : "csv"
    },
    "ISOT Fake News": {
        "name": "ISOT Fake News",
        "columns" : ["title", "text", "label"],
        "domain": "",
        "raw_type" : "csv"
        },
    "FineFake": {
        "name": "FineFake",
        "columns" : ["text", "label"],
        "domain": "",
        "raw_type" : "pkl"
    },
    "FEVER": {
        "name": "FEVER",
        "columns" : ["label", "claim"],
        "domain": "",
        "raw_type" : "jsonl"
    },
    "Fakeddit": {
        "name": "Fakeddit",
        "columns" : ["title", "label"],
        "domain": "",
        "raw_type" : "tsv"
    },
    "Fake News Prediction": {
        "name": "Fake News Prediction",
        "columns" : ["title", "text", "label"],
        "domain": "",
        "raw_type" : "csv"
    },
    "Fake News Corpus": {
        "name": "Fake News Corpus",
        "columns" : ["content", "title", "label"],
        "domain": "",
        "raw_type" : "csv"
    },
    "Climate-FEVER": {
        "name": "Climate-FEVER",
        "columns" : ["claim", "label"],
        "domain": "",
        "raw_type" : "csv"
    }
}