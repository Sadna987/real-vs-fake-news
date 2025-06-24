# 📰 Fake News Detection using Logistic Regression

This project uses a **Logistic Regression** model to classify news articles as **FAKE** or **REAL**. It employs **TF-IDF vectorization** for text processing and includes various visualizations like **confusion matrix**, **word clouds**, and **TF-IDF bar charts** to analyze the results.

## 📁 Dataset

* `Fake.csv` — contains fake news articles.
* `True.csv` — contains real news articles.
* You can find these datasets from sources like Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## 📦 Libraries Used

* `pandas`, `numpy` — for data manipulation
* `matplotlib`, `seaborn` — for plotting
* `sklearn` — for preprocessing and machine learning
* `wordcloud` — for visualizing frequent words

## 🚀 Features

* Preprocessing and merging real/fake datasets
* TF-IDF text vectorization
* Logistic Regression model training
* Model evaluation with accuracy, confusion matrix, and classification report
* Word clouds for FAKE and REAL news
* Top 20 TF-IDF words for both classes

## 🧠 How It Works

1. **Load and Merge Data**
   Load `Fake.csv` and `True.csv`, label them, and shuffle the dataset.

2. **Train-Test Split & TF-IDF Vectorization**
   Convert the text into numeric vectors using `TfidfVectorizer`.

3. **Model Training**
   Use `LogisticRegression` to train on the TF-IDF vectors.

4. **Evaluation & Visualization**

   * Accuracy and classification report
   * Confusion matrix using Seaborn
   * Word clouds for each class
   * Top 20 TF-IDF terms bar chart for both classes

## 🧪 Usage

1. Place `Fake.csv` and `True.csv` in your working directory.

2. Run the script in a Python environment (Jupyter Notebook, VS Code, etc.).

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

3. Update the file paths in the code if needed:

```python
pd.read_csv(r"C:\Path\To\Your\Fake.csv")
pd.read_csv(r"C:\Path\To\Your\True.csv")
```

4. Run all cells to train the model and view the results.

## 📊 Sample Output

* **Accuracy** score on test set.
* **Classification report** with precision, recall, and F1-score.
* **Confusion matrix heatmap**.
* **Word clouds** for FAKE and REAL news articles.
* **Bar charts** of top TF-IDF terms.

