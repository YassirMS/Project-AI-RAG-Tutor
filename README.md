# Projet IA — Tuteur RAG & Classification d'images

Projet IA réalisé à l'**ISIMA** (Université Clermont Auvergne), année 2025–2026.

Le projet combine trois volets : un **agent tuteur IA** basé sur le RAG pour assister le
développement Python, une **IA apprenante** de classification d'images (MLP, CNN, Transfer
Learning), et une analyse **RSE / impact environnemental** via CodeCarbon.

---

## Sommaire

- [Aperçu](#aperçu)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Stack technique](#stack-technique)
- [Auteurs](#auteurs)

---

## Aperçu

### 1. Tuteur IA (RAG)

Un assistant pédagogique qui répond à des questions sur le machine learning et le deep
learning, et aide à corriger/améliorer du code Python (scikit-learn, TensorFlow, PyTorch).
L'architecture **Retrieval-Augmented Generation** ancre les réponses sur une base
documentaire (cours, documentations officielles) pour limiter les hallucinations.

Pipeline : extraction PDF → nettoyage → chunking → embeddings → recherche vectorielle →
reranking → génération.

### 2. IA apprenante (Vision)

Classification d'images sur le dataset **banana-sushi** (4 classes : `banana`, `pizza`,
`sushi`, `tomato`). Trois architectures sont implémentées et comparées :

- **MLP** — baseline
- **CNN** — réseau convolutif entraîné depuis zéro, avec visualisation Grad-CAM
- **Transfer Learning** — ResNet50 pré-entraîné sur ImageNet (feature extraction)

### 3. RSE / Green AI

Mesure de la consommation énergétique et des émissions CO₂ (CPU / GPU / RAM) avec
**CodeCarbon**, sur les différents LLM du tuteur et sur les modèles de vision.

---

## Structure du dépôt

```
.
├── Tuteur_IA.ipynb                          # Agent tuteur RAG
├── MLP_CNN_Transfer_Learning__TensorFlow_.ipynb   # MLP, CNN, TL (TensorFlow/Keras)
├── transfer_learning_pytorch.ipynb          # Transfer Learning ResNet50 (PyTorch)
├── Rapport_Projet_Tuteur_IA.pdf             # Rapport complet
├── data/
│   ├── train/   (banana/ pizza/ sushi/ tomato/)
│   ├── val/
│   └── test/
└── README.md
```

> Le dataset est attendu sous `data/` (compatible `ImageFolder`), ou fourni via
> `data_cleaned.zip` à décompresser à la racine.

---

## Installation

Python 3.10+ recommandé.

**Tuteur IA (RAG)**
```bash
pip install chromadb sentence-transformers transformers accelerate pypdf2 \
            codecarbon langchain-text-splitters torch
```

**IA apprenante — TensorFlow**
```bash
pip install tensorflow opencv-python matplotlib codecarbon numpy
```

**IA apprenante — PyTorch**
```bash
pip install torch torchvision torchsummary numpy matplotlib Pillow
```

Un GPU est fortement recommandé pour l'inférence du LLM et l'entraînement des modèles.

---

## Utilisation

Chaque partie est un notebook autonome, à exécuter cellule par cellule.

- **`Tuteur_IA.ipynb`** — place tes documents PDF dans le dossier prévu, lance
  l'initialisation du RAG puis pose tes questions au tuteur.
- **`MLP_CNN_Transfer_Learning__TensorFlow_.ipynb`** — entraîne et compare les trois
  architectures (sélection via `modeles_a_tester = ["mlp", "cnn", "transfer"]`).
- **`transfer_learning_pytorch.ipynb`** — entraîne le ResNet50 et visualise les courbes
  d'apprentissage.

Assure-toi que `data/` (ou `data_cleaned.zip`) est bien présent avant de lancer la partie
vision.

---

## Résultats

### IA apprenante (sur l'ensemble de validation)

| Modèle                         | Accuracy | F1-Score | CO₂ (mg) |
|--------------------------------|:--------:|:--------:|:--------:|
| MLP (baseline)                 |   67 %   |   0.63   |   0.20   |
| CNN simple                     |   72 %   |   0.68   |   0.22   |
| **Transfer Learning (ResNet50)** | **98 %** | **0.98** |   0.25   |

Le Transfer Learning l'emporte largement : convergence quasi-instantanée et meilleure
précision pour un surcoût énergétique minime. La vraie sobriété, c'est l'efficacité.

### Tuteur IA

Sur 14 questions d'évaluation : **11 réponses pleinement pertinentes (≈ 79 %)**, aucune
réponse non pertinente. La qualité s'améliore nettement quand le RAG est activé.

---

## Stack technique

| Domaine        | Outils |
|----------------|--------|
| Embeddings     | `all-MiniLM-L6-v2` (sentence-transformers) |
| Reranking      | Cross-Encoder `ms-marco-MiniLM-L-6-v2` |
| Vector store   | ChromaDB (indexation HNSW, similarité cosinus) |
| LLM            | Mistral-7B-Instruct *(aussi testés : Qwen2.5, TinyLlama, Llama-3.2)* |
| Deep Learning  | TensorFlow / Keras, PyTorch / torchvision |
| Modèle vision  | ResNet50 (pré-entraîné ImageNet) |
| Empreinte CO₂  | CodeCarbon |

---

## Auteurs

- Rzeigui Ahmed
- Masfour Yassir
- Nasri Ayoub
- Mohamed Amine Rannen

*ISIMA — Institut Supérieur d'Informatique, de Modélisation et de leurs Applications,
Université Clermont Auvergne — 2025/2026.*
