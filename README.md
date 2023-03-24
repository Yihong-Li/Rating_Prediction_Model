# Rating_Prediction_Model
A project for DSC 80 at UCSD
(by: Zixin Wei, Yihong Li)


# Problem Identification
The goal of this project was to build a predictive model that predicts recipes' customer ratings. Our dataset contains recipes and ratings from food.com. We used the popular Python library Sklearn to develop a machine learning model.

Specifically, our group will continue the work on the cleaned recipe and rating data (from project 3 EDA). We will build a classifier which performs multiclass classification. The response variable is the rating of recipes. The variable rating is a categorical ordinal variable which has a score of 1 to 5, where 5 is the highest and 1 is the lowest. We used both accuracy and F1-score to evaluate the model during the project. This is because our data is in fact an imbalanced dataset, and we choose to balance the data by using upsampling and downsampling. Before we balance the dataset, our primary concern is the F-1 score since it better evaluates the model's performance on imbalanced data.
Nevertheless, we also pay some attention to the accuracy.



# Baseline Model

## The Issue of Imbalanced Data
Following the same EDA process as we did earlier (available at https://yihong-li.github.io/Recipe-Analysis-EDA/), we cleaned the original dataset. We also dropped null values in our dataset. After that, our entire dataset contains 217806 examples. To evaluate our model's performance on unseen data, we splitted the dataset into training set and validation set using a ratio of 4:1. For our simplest baseline model, we trained our model using two features: 'review' and 'n_step'. We built a pipeline that transformed our 'review' to unigram bag of words and made prediction by a multinomial naive bayes classifier. After fitting the baseline model, we achieved an test accuracy of around 77%. Though initially this result sames satisfactory, when we manually examined the predictions of our training data, we discovered that almost all of the predictions are 5. It turned out that the training data we used is severely imbalanced. Specifically, 77.3% of the examples have rating 5, 17.0% have rating 4, 3.2% have rating 3, 1.1% have rating 2, and 1.3% have rating 1. We believed that when traing the model on this imbalanced data, our model collapsed and might find a shortcut to always predict 5. So we switched from accuracy to F1 score to evaluate our model, and the F1 score was 0.3377, which is very low (note that since our balanced training set is randomly sampled from the unbalanced training set, the exact F1 score and accuracy will vary a little bit each time).


## Creating balanced training data by upsampling and downsampling
To solve the unbalanced dataset, we decided to combine upsampling and downsampling techniques to create a balanced dataset. We sampled a total 100,000 samples from the unbalanced training dataset, where each rating has 20,000 samples. In this way, we downsample the rating 5 and upsample other ratings. The resulting distribution of ratings are as follows. (attach the image)


## Training the Baseline Model
For the baseline model, we decided to include “n_steps” and “reviews” as two features to predict the ratings of the recipe. The feature “n_step” is quantitative, which shows how many steps are included in the recipes. Another feature “reviews” is textual data, which are texts showing customers’ feedback. To convert reviews into quantitative values, we used “countvectorizer” to perform the “bag of words'' encoding. For the classifier, we choose to use the Multinomial Naive Bayes classifier (multinomialNB), because this classifier is suitable for discrete features such as reviews. The baseline model achieves an accuracy of 0.73 for training data and 0.64 for test data. It has a F1-score of 0.3914. We believe the baseline model is not good, because both the accuracy for test data and F1-score are low, which indicates the poor performance of the predictions.


# Final Model
For the final model, we added two features: minutes and n_ingredients. Both of these two features are quantitative features. Minutes shows the amount of time needed for the recipe and n_ingredients shows the number of ingredients needed for the recipe. We believe these two features can improve the prediction task, because recipes that take a big amount of time and numerous ingredients may tend to have lower ratings, as people tend to prefer those easier recipes. 

We then notice that the classifier we chose, MultinomialNB, assumes the features are discrete features and their values are counts. Based on this knowledge, we use kbinsdiscretizer to discretize “n_steps”, “minutes”, and “n_ingredients”, which decreases the unique values of these three features and increases the model accuracy by approximately 3%.

Our major improvement in pipeline was attributed to changing the CountVectorizer's ngram_range hyperparameter from default unigram to unigram, bigram, and trigram. We believed that a unigram model was too simple to capture the complex contextual meaning of the review sentences. Specifically, we manually tuning three hyperparamters: ngram_range, max_df, and min_df. ngram_range controlls the complexity of our model as larger n generally contains richer semantic information in text and can better identify pattern in textual data. Max_df set the threshold to ignore terms that have a document frequency strictly higher than that threshold. We believed that the most frequent features in text (eg. 'the', 'I think', etc) didn't contain useful semantic information and it's useful to remove them. Min_df set the threshold that ignore terms that have a document frequency strictly lower than the threshold. We tuned this hyperparameter because it can reduce overfitting in our model.

Unlike using grid search to optimize the hyperparameters. We performed manual search mainly because the training set is too large and grid search takes forever to run. After a series of manual experiment on different combination of hyperparameters, we concluded that ngram_range=(1, 3), max_df=10000, min_df=2 achieved the best validation f1-score. After retraining the model on the entire training set and validation set using the best hyperparameters, we saved our model and tested it's performance on the test set. Our final model's f1-score was 0.5064, which is roughly 0.11 points higher than our baseline and 0.16 points higher than the model we trained at the very beginning on a severely imbalanced dataset. Moreover, we achieved on accuray of 0.7873, which is also significantly higher than any previous models.


# Fairness Analysis




