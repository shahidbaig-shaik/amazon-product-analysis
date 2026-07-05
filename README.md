# Amazon Product Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![NLP](https://img.shields.io/badge/NLP-Sentiment-blueviolet) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit) ![License](https://img.shields.io/badge/license-MIT-green)

> NLP pipeline that transforms Amazon product reviews into sentiment scores, rating predictions, and category-level business insights.

## Overview

Customer reviews are one of the richest — and most underutilized — signals in e-commerce. This project processes Amazon product review data through a full NLP pipeline: sentiment classification, star-rating prediction, and aggregated category-level insights. A Streamlit dashboard and a chat interface (`chat.py`) allow business users to query insights without writing code.

## Tech Stack

| Component | Technology |
|---|---|
| NLP / Sentiment | scikit-learn text pipeline (TF-IDF + classifier) |
| Rating Prediction | Regression / classification on review features |
| Data Processing | pandas |
| Insights Storage | `insights.json` (pre-computed category summaries) |
| Dashboard | Streamlit (`app.py`) |
| Chat Interface | `chat.py` (natural language query layer) |

## How It Works

- **Ingest** Amazon review dataset (text, star rating, product category)
- **Classify** review sentiment (positive / negative / neutral) via trained NLP model
- **Predict** star ratings from review text using regression pipeline
- **Aggregate** category-level sentiment trends and surfaced pain points
- **Serve** insights via Streamlit dashboard and conversational chat interface

## Quick Start

```bash
git clone https://github.com/shahidbaig-shaik/amazon-product-analysis
cd amazon-product-analysis
pip install -r requirements.txt
streamlit run app.py
```
