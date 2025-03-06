# Web-Scraping

# Song Lyrics Artist Classification

## Overview
This project focuses on building a **text classification model** that predicts the artist based on song lyrics. Using **machine learning** and **natural language processing (NLP)** techniques, we will collect a dataset of song lyrics, preprocess the text, and train a classifier to identify the artist.

---

## Objectives
- **Scrape song lyrics** from web pages to create a dataset.
- **Extract hyperlinks** leading to individual song pages.
- **Download and clean song lyrics** to prepare training data.
- **Vectorize lyrics** using the **Bag of Words** method.
- **Train a classification model** to predict the artist based on lyrics.

---

## Tech Stack
- **Python** - Main programming language
- **Requests & BeautifulSoup** - Web scraping tools
- **NLTK / Scikit-learn** - NLP and machine learning libraries
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization

---

## Dataset Collection
### 1. **Download an HTML Page with Song Links**
- Use the `requests` library to fetch an HTML page containing links to multiple song lyrics.
- Example:
  ```python
  import requests
  response = requests.get("https://example.com/song-lyrics")
  html_content = response.text
  ```

### 2. **Extract Hyperlinks to Song Pages**
- Use `BeautifulSoup` to parse the HTML and extract links to song lyrics pages.
- Example:
  ```python
  from bs4 import BeautifulSoup
  soup = BeautifulSoup(html_content, 'html.parser')
  song_links = [a['href'] for a in soup.find_all('a', href=True) if 'lyrics' in a['href']]
  ```

### 3. **Download and Extract Song Lyrics**
- Loop through extracted URLs and fetch lyrics.
- Clean the text by removing special characters and unnecessary formatting.

---

## Text Preprocessing & Feature Engineering
### 4. **Vectorize Lyrics Using Bag of Words**
- Convert raw text into numerical features using **CountVectorizer** from `scikit-learn`.
- Example:
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(lyrics)
  ```

---

## Model Training & Evaluation
### 5. **Train a Classification Model**
- Use machine learning algorithms like **Logistic Regression**, **Naïve Bayes**, or **Random Forest** to train the model.
- Example:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import MultinomialNB

  X_train, X_test, y_train, y_test = train_test_split(X, artists, test_size=0.2, random_state=42)
  model = MultinomialNB()
  model.fit(X_train, y_train)
  ```

### 6. **Evaluate Model Performance**
- Measure model accuracy using **precision, recall, and F1-score**.
- Example:
  ```python
  from sklearn.metrics import classification_report
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
  ```
  
---

## Results & Learnings
- Successfully scraped song lyrics from multiple web pages.
- Processed and vectorized text using **Bag of Words**.
- Trained a classifier to **predict the artist from lyrics**.
- Understood the importance of **data preprocessing and feature extraction** in NLP.

---

## License
This project is licensed under the MIT License.
