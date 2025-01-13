!pip install pyspark

import time
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pow, lit

# Initialize SparkSession
spark = SparkSession.builder.appName('recommendation').getOrCreate()

# Load datasets
movies = spark.read.load("/content/movies.csv", format='csv', header=True)
ratings = spark.read.load("/content/ratings.csv", format='csv', header=True)
links = spark.read.load("/content/links.csv", format='csv', header=True)
tags = spark.read.load("/content/tags.csv", format='csv', header=True)

# Display ratings dataset
ratings.show()

# Cast columns to appropriate types
df = ratings.withColumn('userId', ratings['userId'].cast('int')). \
    withColumn('movieId', ratings['movieId'].cast('int')). \
    withColumn('rating', ratings['rating'].cast('float'))

df.printSchema()

# Split the data into train, validation, and test sets
train, validation, test = df.randomSplit([0.6, 0.2, 0.2], seed=0)
print("The number of ratings in each set: {}, {}, {}".format(train.count(), validation.count(), test.count()))

# Define RMSE calculation
def RMSE(predictions):
    squared_diff = predictions.withColumn("squared_diff", pow(col("rating") - col("prediction"), 2))
    mse = squared_diff.selectExpr("mean(squared_diff) as mse").first().mse
    return mse ** 0.5

# Perform Grid Search for ALS
from pyspark.ml.recommendation import ALS

def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None

    # Run Grid Search for all the parameters
    for n in n_factors:
        for reg in reg_param:
            als = ALS(rank=n,
                      maxIter=num_iterations,
                      seed=0,
                      regParam=reg,
                      userCol="userId",
                      itemCol="movieId",
                      ratingCol="rating",
                      coldStartStrategy="drop")
            model = als.fit(train)
            predictions = model.transform(valid)
            rmse = RMSE(predictions)
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(n, reg, rmse))

            # Track the best model using RMSE
            if rmse < min_rmse:
                min_rmse = rmse
                best_n = n
                best_reg = reg
                best_model = model

    pred = best_model.transform(train)
    train_rmse = RMSE(pred)

    # Print the best model and its metrics
    print('\nThe best model has {} latent factors and regularization = {}:'.format(best_n, best_reg))
    print('training RMSE is {}; validation RMSE is {}'.format(train_rmse, min_rmse))
    return best_model

# Define hyperparameters
num_iterations = 10
ranks = [6, 8, 10, 12]
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]

# Run Grid Search
final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)

# Evaluate on the test set
pred_test = final_model.transform(test)
print('The testing RMSE is ' + str(RMSE(pred_test)))

# Generate recommendations for a single user
user_id = 25
single_user = test.filter(test['userId'] == user_id).select(['movieId', 'userId'])
single_user.show()

# Display movies the user has liked
print("Movies liked by user with ID", user_id)
single_user_ratings = test.filter(test['userId'] == user_id).select(['movieId', 'userId', 'rating'])
single_user_ratings.join(movies, 'movieId').select('movieId', 'title', 'rating').show()

# Generate recommendations for the user
all_movies = df.select('movieId').distinct()
user_movies = single_user_ratings.select('movieId').distinct()
movies_to_recommend = all_movies.subtract(user_movies)

# Predict ratings for movies the user has not rated yet
recommendations = final_model.transform(movies_to_recommend.withColumn('userId', lit(user_id)))

# Filter out movies with negative predictions
recommendations = recommendations.filter(col('prediction') > 0)

# Display recommendations with movie names
recommended_movies = recommendations.join(movies, 'movieId').select('movieId', 'title', 'prediction')
ordered_recommendations = recommended_movies.orderBy(col('prediction').desc())
ordered_recommendations.show()
