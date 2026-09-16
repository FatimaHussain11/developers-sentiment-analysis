# Sentiment Analysis of AI Software Development: A Transfer Learning Approach Using DistilBERT

<!-- Add a project banner / architecture diagram / screenshot here -->
![Project Banner](path/to/your/image.png)

## 📌 Overview

This project analyzes developer and user sentiment toward AI-based software development tools (e.g., Quin, CLI, Gemini, Codex) using a fine-tuned **DistilBERT** transformer model. Comments collected from **Reddit** and **Quora** are classified into three sentiment classes — **Positive**, **Negative**, and **Neutral** — enabling data-driven insight into developer satisfaction and software quality.

The trained model is deployed as a **local Streamlit application** for real-time sentiment classification.


## 🎯 Objectives

- Understand how developer sentiment reflects software quality, project progress, and development challenges.
- Support data-driven decision-making in AI software development.
- Contribute to more intelligent, emotion-aware software systems.

## 📂 Dataset

- **Size:** 3,318 annotated comments (paper text also references 3,310 in the abstract)
- **Sources:** Reddit and Quora
- **Classes:** Positive, Negative, Neutral (balanced to minimize classification bias)

## 🧠 Model Architecture

- **Base model:** DistilBERT — a distilled version of BERT retaining ~97% of its language understanding while being smaller and faster.
- **Tokenizer:** DistilBERT tokenizer (handles sub-word representations for technical language).

## ⚙️ Data Preprocessing

- Removal of URLs, special characters, and HTML tags
- Conversion of textual labels into numerical values
- Tokenization via the DistilBERT tokenizer

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Library | Hugging Face Transformers |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Environment | GPU-enabled |

## 📊 Results

The fine-tuned model achieved strong accuracy and F1-score across all three sentiment categories on the held-out test set.

## 🖥️ Application


<!-- Add a screenshot of the Streamlit app here -->
![App Screenshot](path/to/app-screenshot.png)
<img width="510" height="689" alt="c" src="https://github.com/user-attachments/assets/66f654fd-d8d0-4f9d-8e37-1c3bce53e373" />


## 📁 Suggested Project Structure

```
├── data/               # Raw and processed comment datasets
├── model/              # Fine-tuned DistilBERT model artifacts
├── requirements.txt    # Python dependencies
└── README.md
```

## 🔧 Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

## 📚 References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT: A distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*, 216–225.
- Baccianella, S., Esuli, A., & Sebastiani, F. (2010). SentiWordNet 3.0: An enhanced lexical resource for sentiment analysis and opinion mining. *LREC*.
- Zhang, H., Kou, C., Wang, X., & Li, X. (2020). Sentiment analysis for software engineering: How far can we go? *IEEE Transactions on Software Engineering*, 46(9), 1010–1031.
- Jongeling, R., Datta, S., Serebrenik, A., & van Deursen, A. (2017). On negative results when using sentiment analysis tools for software engineering research. *Empirical Software Engineering*, 22(5), 2543–2584.
