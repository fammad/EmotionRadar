# Emotion Radar

Multi-label emotion classification on Reddit comments, built as the final project for **DM1590: Machine Learning for Media Technology** at KTH.

Emotion Radar reads a short piece of text and tags it with any of seven emotions at once, then shows *which words* drove each prediction. It pairs a working classifier with an honest investigation into where the approach breaks down and why.

```
"I'm furious but also kind of relieved it's over."
   -> anger  (driven by: furious)
   -> joy    (driven by: relieved)
```

---

## What this project actually is

Most student ML projects stop at "we trained a model and got X% accuracy." This one does that, but the more interesting half is a question we set out to answer first:

**Do emotional comments naturally cluster by emotion, before any labels are involved?**

The answer turned out to be no, and proving *that* cleanly took more thought than the classifier did. The project is organised around two complementary halves:

1. **Unsupervised** (run first): can structure emerge on its own? We test this with K-Means and DBSCAN on a TF-IDF/LSA representation, and treat the result as a finding to be defended rather than a number to be maximised.
2. **Supervised**: given labels, how well can we actually predict the seven emotions? We compare a Naive Bayes baseline against a balanced logistic regression, and evaluate the winner from several angles.

The dataset is [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Demszky et al., 2020), ~58k Reddit comments. Its 28 fine-grained labels are remapped to the 7 [Ekman basic emotions](https://en.wikipedia.org/wiki/Emotion_classification#Basic_emotions) (anger, disgust, fear, joy, sadness, surprise, neutral) for more samples per class and cleaner interpretation.

---

## Headline results

| | Naive Bayes (baseline) | Logistic Regression (balanced) |
|---|:---:|:---:|
| Macro-F1 | 0.25 | **0.51** |
| Micro-F1 | 0.44 | **0.58** |
| Macro ROC-AUC | 0.75 | **0.84** |

Logistic regression roughly doubles the macro-F1 of the baseline. The gain comes almost entirely from `class_weight="balanced"`, which forces the model to take the rare emotions (disgust and fear, ~100 training examples each) seriously instead of ignoring them.

Performance is uneven by design rather than by accident. **Joy** is detected reliably (F1 ≈ 0.75) because it has thousands of examples and a distinctive vocabulary. **Fear** scores surprisingly well (F1 ≈ 0.53) on only 98 samples, because its vocabulary ("scared", "nervous", "afraid") is narrow and consistent. The vaguer, rarer classes lag, and the report explains each case rather than averaging it away.

### The unsupervised finding

Three independent checks agree that emotions do **not** form natural groups in TF-IDF space:

- **K-Means** at k=7: silhouette ≈ 0.05, ARI ≈ 0, NMI ≈ 0.05. One cluster absorbs most of the data.
- **DBSCAN** eps sweep: no setting produces several balanced clusters. Small `eps` labels nearly everything as noise; large `eps` collapses everything into one blob. There is no middle ground.
- A **300-dimension control**: tripling the retained variance barely moves the silhouette, which rules out aggressive dimensionality reduction as the cause.

The conclusion is that the limitation lives in the *representation*. Bag-of-words TF-IDF encodes which words appear, not what they mean, so two angry comments with no shared vocabulary sit as far apart as an angry one and a happy one. This is the kind of result that is easy to mistake for a bug; the project's contribution is showing, with evidence, that it is a real property of the feature space.

---

## Why TF-IDF instead of modern embeddings?

A reasonable question, since sentence embeddings (e.g. Sentence-BERT) would almost certainly cluster better. The choice was deliberate: TF-IDF features *are individual words*, so the logistic regression coefficients map directly back to vocabulary. That is what powers the word-level explanations in the demo. We traded raw performance for interpretability, on purpose, and the report names this as a trade-off rather than hiding it. The "Future work" section lays out the embedding-based version as the natural next step.

---

## Repository structure

```
EmotionRadar/
├── EmotionRadar_Report_DM1590_Group15.ipynb   # main deliverable: report + collapsible code
├── data/
│   ├── processed/         # train.csv, val.csv, test.csv (Ekman-remapped)
│   └── models/            # nb_model.pkl, lr_model.pkl, tfidf.pkl
├── report/
│   └── figures/           # generated plots
└── README.md
```

The notebook is written to read **top to bottom as a report**, with code collapsed by default and expandable for inspection.

---

## Running it

```bash
git clone https://github.com/fammad/EmotionRadar.git
cd EmotionRadar
pip install -r requirements.txt
jupyter lab EmotionRadar_Report_DM1590_Group15.ipynb
```

Then run all cells. On first run, if `data/processed/` is empty, the notebook downloads GoEmotions from HuggingFace and rebuilds the splits automatically; afterwards it loads the cached CSVs.

**Core dependencies:** `scikit-learn`, `nltk`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `datasets`.

---

## Method summary

| Stage | Choice | Notes |
|---|---|---|
| Labels | 28 -> 7 Ekman emotions | multi-label; a comment can carry several |
| Features | TF-IDF, 1–2 grams, 5k vocab | fit on train only, no leakage |
| Reduction | TruncatedSVD (LSA) to 100 dims | 300-dim version kept as a control |
| Unsupervised | K-Means (course) + DBSCAN (new) | t-SNE for visualization only |
| Supervised | MultinomialNB (course) + LogReg balanced (new) | `OneVsRest`, tuned on validation |
| Evaluation | macro/micro-F1, ROC-AUC, PR curves, per-emotion confusion, length stratification, error analysis | macro-F1 as the headline metric |

A note on t-SNE: it is used purely to project the embeddings to 2-D for plotting. It is not a clustering method and feeds nothing downstream.

---

## Reproducibility

- A single fixed seed (`RANDOM_STATE = 42`) governs every stochastic step: the split, K-Means initialisation, t-SNE, and sampling.
- The TF-IDF vectoriser is fit on training data only; validation and test reuse the same vocabulary.
- Trained models and the fitted vectoriser are saved to `data/models/` so results can be reloaded without retraining.

---

## What could be improved for next step

- Per-emotion decision thresholds tuned on validation, instead of a shared 0.5.
- A Sentence-BERT pipeline to measure exactly how much clustering and accuracy improve once interpretability is traded away.
- A fine-tuned DistilBERT baseline, to see how far a context-aware model closes the gap on sarcasm and negation, the two failure modes TF-IDF cannot handle.

---

## References

- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020.
- Ekman, P. (1992). *An Argument for Basic Emotions.* Cognition & Emotion, 6(3–4), 169–200.
- Géron, A. *Hands-On Machine Learning with Scikit-Learn and PyTorch.* O'Reilly.
- scikit-learn documentation: multiclass/multilabel classification, clustering, and TruncatedSVD.

---

## Team

Group 15, DM1590, KTH Royal Institute of Technology — Fuad Mammadov, Qingya Li, Xinyutong Zhang, Jintong Jiang. May 2025.

*Project graded against an A–F rubric covering supervised application, unsupervised application, evaluation, and presentation.*