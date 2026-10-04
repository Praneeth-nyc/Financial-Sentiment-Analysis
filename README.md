# Financial Sentiment Analysis

A deep learning-based NLP project that classifies financial text into **Positive, Negative, and Neutral** sentiment categories using **SimpleRNN and LSTM** models.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* NLP
* SimpleRNN
* LSTM

## Project Overview

The objective of this project is to analyze financial text and automatically determine its sentiment.

The project uses the **Financial Sentiment Analysis** dataset and performs text preprocessing, tokenization, sequence padding, and deep learning-based classification.

Two neural network architectures were implemented:

* SimpleRNN
* LSTM

The models classify each financial sentence into one of three sentiment classes:

| Label | Sentiment |
| ----- | --------- |
| 0     | Neutral   |
| 1     | Positive  |
| 2     | Negative  |

## Workflow

```text
Financial Text
       ↓
Data Cleaning
       ↓
Text Tokenization
       ↓
Sequence Conversion
       ↓
Padding
       ↓
Word Embedding
       ↓
SimpleRNN / LSTM
       ↓
Softmax Classification
       ↓
Neutral / Positive / Negative
```

## Dataset

The project uses the **Financial Sentiment Analysis** dataset available through Kaggle.

The dataset is downloaded programmatically using `kagglehub`, so the dataset itself is not stored in this repository.

The dataset contains financial sentences along with their corresponding sentiment labels.

## Data Preprocessing

The following preprocessing steps were performed:

1. Converted text to lowercase.
2. Removed special characters.
3. Removed unnecessary whitespace.
4. Tokenized the sentences.
5. Converted words into numerical sequences.
6. Used an `<OOV>` token for unknown words.
7. Padded sequences to a fixed maximum length of 150.

Example:

```text
Original:
"We gained profit!"

Cleaned:
"we gained profit"
```

## Exploratory Data Analysis

The project includes analysis of:

* Sentiment class distribution
* Number of positive, negative, and neutral sentences
* Sentence length
* Word count

Class percentages were also calculated to understand the distribution of the dataset.

## Model Architecture

### SimpleRNN

The SimpleRNN model consists of:

```text
Embedding
    ↓
SimpleRNN (64 units)
    ↓
Dropout (0.5)
    ↓
Dense (64)
    ↓
Dense (32)
    ↓
Dense (16)
    ↓
Softmax Output (3 classes)
```

### LSTM

The LSTM model consists of:

```text
Embedding
    ↓
LSTM (64 units)
    ↓
Dropout (0.5)
    ↓
Dense (64)
    ↓
Dense (32)
    ↓
Dense (16)
    ↓
Softmax Output (3 classes)
```

## Handling Class Imbalance

Class weights were calculated using `compute_class_weight` from Scikit-learn.

These weights were passed during model training so that underrepresented sentiment classes receive greater importance during training.

## Training Configuration

The models were trained using:

* Optimizer: Adam
* Learning Rate: `0.001`
* Batch Size: `32`
* Epochs: `8`
* Loss Function: Sparse Categorical Crossentropy
* Validation Split: `20%`
* Train/Test Split: `75% / 25%`

## Model Files

The trained models and preprocessing objects are stored in the `Models` directory:

```text
Models/
├── lstm_model.keras
├── rnn_model.keras
├── tokenizer.pkl
└── max_len.pkl
```

* `lstm_model.keras` — trained LSTM model
* `rnn_model.keras` — trained SimpleRNN model
* `tokenizer.pkl` — fitted text tokenizer
* `max_len.pkl` — maximum sequence length used for padding

## Prediction

The trained LSTM model can be used to predict the sentiment of new financial text.

Example:

```python
predict_sentiment("we gained profit")
```

The input text is first cleaned and converted into a numerical sequence using the saved tokenizer. The sequence is then padded and passed to the LSTM model.

The model returns the predicted sentiment:

```text
Prediction: [...]
Sentiment: Positive
```

## Project Structure

```text
financial-sentiment-analysis/
│
├── Models/
│   ├── lstm_model.keras
│   ├── rnn_model.keras
│   ├── tokenizer.pkl
│   └── max_len.pkl
│
├── Notebooks/
│   └── finance.ipynb
│
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Praneeth-nyc/financial-sentiment-analysis.git
```

### 2. Navigate to the project

```bash
cd financial-sentiment-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

Open:

```text
Notebooks/finance.ipynb
```

Run the notebook cells sequentially to:

* Download the dataset
* Perform data preprocessing
* Analyze the dataset
* Train the RNN model
* Train the LSTM model
* Save the trained models
* Perform sentiment prediction

## Future Improvements

* Compare additional transformer-based models such as BERT.
* Perform hyperparameter tuning.
* Add precision, recall, F1-score, and confusion matrix visualizations.
* Develop a web interface for real-time sentiment prediction.
* Deploy the trained model as an API.

**Praneeth Aitharaju**

GitHub: [Praneeth-nyc](https://github.com/Praneeth-nyc)
