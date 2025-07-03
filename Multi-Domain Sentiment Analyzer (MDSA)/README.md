## Multi-Domain Sentiment Analyzer (MDSA)

The **Multi-Domain Sentiment Analyzer (MDSA)** is a powerful and versatile project that demonstrates real-world use of NLP and Hugging Face Transformers.

---

## ✅ **1. Project Goal**

Create a universal sentiment classifier that performs well across **product reviews**, **tweets**, and **financial news** using **pre-trained transformers** like `BERT`, `RoBERTa`, or `DistilBERT`.

---

## 📁 **2. Datasets (Multi-Domain)**

| Domain         | Dataset                                                                      | Hugging Face Dataset Name |
| -------------- | ---------------------------------------------------------------------------- | ------------------------- |
| Product Review | [Amazon Polarity](https://huggingface.co/datasets/mteb/amazon_polarity)           | `amazon_polarity`         |
| Twitter        | [Sentiment140](https://huggingface.co/datasets/contemmcm/sentiment140)                 | `sentiment140`            |
| Finance        | [Financial PhraseBank](https://huggingface.co/datasets/atrost/financial_phrasebank/viewer) | `financial_phrasebank`    |

---

## 🤗 **3. Transformers to Try**

* `bert-base-uncased` (baseline)
* `distilbert-base-uncased` (faster)
* `roberta-base` (robust performance)
* `bertweet-base` (for Twitter data)

---

## 🏗️ **4. Architecture Overview**

```plaintext
Dataset Loader → Preprocessing (Tokenization) →
Multi-Domain DataLoader →
Fine-tuning Transformer →
Evaluation Metrics →
UI Interface (Gradio or Streamlit) →
Deployment (optional)
```