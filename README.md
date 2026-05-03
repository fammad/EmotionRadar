# EmotionRadar
Multi-label emotion detection from text using supervised and unsupervised machine learning.

**Team:** 
**Course:** DM1590 — Machine Learning for Media Technology, KTH

## How to run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download GoEmotions raw data from:
   https://github.com/google-research/google-research/tree/master/goemotions/data
   Place goemotions_1.csv, goemotions_2.csv, goemotions_3.csv into `data/raw/`
4. Run notebooks in order (see /notebooks)

## Folder structure

emotion-radar/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/       
└── report/