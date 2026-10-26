# Emotion Classifier

An NLP-based text emotion classification project. The training notebook prepares labeled text, converts it into numerical features with `CountVectorizer`, compares two scikit-learn classifiers, and reports their test-set accuracy. The Streamlit app uses the saved model artifacts to classify text entered by a user.

## Project structure

```text
emotion-classifier/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── sentiment.ipynb
└── models/
    ├── sentiment_model.pkl
    ├── count_vectorizer.pkl
    └── emotion_mapping.pkl
```

## Model development

The notebook reads a semicolon-separated file named `train.txt` with two columns: `text` and `emotion`. The displayed dataset contains **16,000 labeled text examples**.

The notebook's preprocessing steps are:

1. Convert text to lowercase.
2. Remove punctuation.
3. Remove numeric characters.
4. Remove non-ASCII characters (which also removes emoji characters).
5. Remove English stopwords after tokenizing with NLTK.
6. Convert emotion labels to integer IDs.

The data is split into training and test sets with an **80/20 split** and `random_state=42`. `CountVectorizer` is fitted on the training text and then used to transform the test text.

### Models evaluated

| Model | Test accuracy reported by the notebook |
|---|---:|
| Logistic Regression | **88.875%** |
| Multinomial Naive Bayes | 76.78125% |

These are the accuracy values printed by the notebook for its test split. They should not be interpreted as a guarantee of performance on new or real-world text. In the notebook's experiment, Logistic Regression achieved the higher accuracy.

## Run the web app locally

1. Install Python if it is not already installed.
2. Open a terminal in the project root directory (the folder containing `app.py`).
3. Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

The app expects the following saved artifacts at these paths:

- `models/sentiment_model.pkl`
- `models/count_vectorizer.pkl`
- `models/emotion_mapping.pkl`

Keep the artifact filenames and folder structure consistent with the paths used in `app.py`.

## Reproduce the notebook experiment

The notebook expects a file named `train.txt` and loads it with a semicolon (`;`) separator, without a header row. The dataset file was not present alongside the notebook I reviewed, and the notebook itself does not identify the dataset's original URL or license. To make the experiment reproducible, provide the correct dataset source and licensing information, then place the permitted `train.txt` file in the notebook's working directory before running the cells.

## Current scope and limitations

- This is a text-classification project based on the dataset and preprocessing in the included notebook.
- The notebook compares Logistic Regression and Multinomial Naive Bayes; the reported accuracy values are from its existing outputs.
- The notebook creates numeric IDs from the order in which emotion labels first appear in the input data. Keep `emotion_mapping.pkl` paired with the corresponding trained model so predicted IDs are interpreted correctly.
- The dataset source and license are not specified in the notebook, so they should be documented before redistributing the dataset or presenting the project as fully reproducible.

## Technologies

- Python
- pandas and NumPy
- NLTK
- scikit-learn
- joblib
- Streamlit
