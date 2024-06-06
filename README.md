# Movie-genres-classification
This project focuses on tagging movie plot synopses with relevant tags using Natural Language Processing (NLP) and machine learning techniques. The goal is to preprocess the text data, extract features, and train multiple machine learning models to predict the most relevant tags for each synopsis.

You can check the operation of the model using a [telegram bot](https://t.me/movie_genres_bot)


### Presentation:

### Dataset:
[dataset kaggle](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?select=partition.json)

### Model Training

Three machine learning models are trained using the preprocessed data:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**

Each model is tuned using GridSearchCV to find the best hyperparameters.

### Metrics:
The trained models are evaluated using F1 Score on a test set.

### Results
The results of the models are summarized in a table, showing the best parameters, F1 Score and Accuracy for each model. 
The best-performing model is selected based on the highest F1 Score.

