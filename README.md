# Twitter Sentiment Analysis

A machine learning project that classifies the sentiment of tweets as positive or negative, comparing deep learning (RNN, LSTM) and classical (Naïve Bayes) approaches on 1.6 million real-world tweets.

**Authors**: Saydain Sheikh, Tahreem Arif, Brendon Weiskott

## Problem Statement

- The focus of this project is to develop a sentiment analysis model for tweets.
- Sentiment analysis plays a crucial role in understanding public opinion and can be applied across domains including business, politics, and social media monitoring.
- The goal is to build a model that accurately classifies tweets into **positive** and **negative** sentiments.

## Data Description

This project uses the **Sentiment140** dataset, which contains **1,600,000 tweets** extracted using the Twitter API, annotated as `0 = negative` and `4 = positive`.

The dataset contains 6 fields:

| Field | Description |
|---|---|
| `target` | Polarity of the tweet (0 = negative, 4 = positive) |
| `ids` | The tweet's ID |
| `date` | Date the tweet was posted |
| `flag` | The query used to collect the tweet (`NO_QUERY` if none) |
| `user` | The user who posted the tweet |
| `text` | The tweet text |

## Implementation

### Data Preprocessing
Preprocessing was performed using the **SpaCy** library, including:
- Stop word removal
- URL and `@mention` removal
- Removal of HTML character entities (e.g. `&quot`)
- Tokenization

Word clouds generated before and after preprocessing showed a cleaner, more sentiment-relevant vocabulary post-cleaning — with common but low-signal terms and residual noise from mentions/links stripped out, surfacing more meaningful high-frequency words for both positive and negative classes.

### Exploratory Data Analysis (EDA)
- **Top words by sentiment**: Frequency analysis of the top 20 words for each class showed positive tweets dominated by words like *good, day, thank, love, go, like*, while negative tweets were dominated by words like *go, work, get, day, miss, not, like, want*.
- **Text length distribution**: Tweet (tokenized) text length was right-skewed for both sentiment classes, with most tweets falling in the 2–10 token range after cleaning.
- **Text length vs. sentiment**: A boxplot comparison showed negative tweets tend to have a slightly wider spread and higher median text length than positive tweets.

## ML Modeling

Three models were built and compared for binary sentiment classification:

### RNN (Recurrent Neural Network)
- Embedding size of 20
- RNN layer with 15 units
- Global Max Pooling
- Dense layer with 32 units and ReLU activation
- Dense output layer with 1 unit and Sigmoid activation (binary classification)
- Adam optimizer, Binary Cross-Entropy loss

### LSTM (Long Short-Term Memory)
- Embedding size of 20
- LSTM layer with 15 units
- Global Max Pooling
- Dense layer with 32 units and ReLU activation
- Dense output layer with 1 unit and Sigmoid activation (binary classification)
- Adam optimizer, Binary Cross-Entropy loss

### Naïve Bayes Classifier
- Hyperparameter tuning identified **alpha = 10** as the best smoothing parameter.
- ROC curve AUC of **0.82**; Precision-Recall curve AUC of **0.81**.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **LSTM** | 0.79 | 0.78 | **0.80** | 0.79 |
| **RNN** | 0.79 | 0.78 | 0.79 | 0.79 |
| Naïve Bayes | 0.74 | 0.77 | 0.69 | 0.73 |

Both neural network models (RNN and LSTM) performed nearly identically and clearly outperformed the Naïve Bayes classifier across all metrics, particularly recall — meaning they were better at correctly identifying true sentiment classes rather than defaulting to the majority prediction.

## Conclusion

- The neural network models (RNN and LSTM) outperformed the Naïve Bayes model across all evaluation metrics.
- Given the inherent difficulty of sentiment analysis on short, informal text like tweets, achieving accuracy/F1 scores in the high 0.7s to 0.8 range represents a reasonably strong result.
- The large vocabulary size and infrequency of many words (due to informal language, slang, and typos common on Twitter) made the classification task more challenging.
- Overall, the results are reasonably strong given these constraints, with room for further improvement through techniques like pretrained word embeddings (e.g. GloVe, Word2Vec) or transformer-based models.

## Author
**Saydain Sheikh, Tahreem Arif, Brendon Weiskott**
Twitter Sentiment Analysis — Team Project
