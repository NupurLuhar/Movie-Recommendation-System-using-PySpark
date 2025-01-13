!pip install pyspark

Collecting pyspark
  Downloading pyspark-3.5.3.tar.gz (317.3 MB)
     |████████████████████████████████| 317.3/317.3 MB 5.4 MB/s eta
0:00:00
  Preparing metadata (setup.py) ... ent already satisfied: py4j==0.10.9.7 /in
/usr/local/lib/python3.10/dist-packages (from pyspark) (0.10.9.7)
  Building wheels for collected packages: pyspark
  Building wheel for pyspark (setup.py) ... e=pyspark-3.5.3-py2.py3-none-any.whl size=317840625
sha256=079014f74a21dba91135353b1afd3405fae30bfaf77b55a62163ab80bceafb3
  Stored in directory: 
/root/.cache/pip/wheels/1b/3a/92/28b93e2bfdbdb07509ca4d6f50c5e407f48dc
e4ddba69a4ab
Successfully built pyspark
Installing collected packages: pyspark
Successfully installed pyspark-3.5.3

import time
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('recommendation').getOrCreate()

movies = spark.read.load("/content/movies.csv", format='csv', header = True)
ratings = spark.read.load("/content/ratings.csv", format='csv', header = True)
links = spark.read.load("/content/links.csv", format='csv', header = True)
tags = spark.read.load("/content/tags.csv", format='csv', header = True)
ratings.show()

+------+-------+------+---------+
|userId|movieId|rating|timestamp|
+------+-------+------+---------+
|     1|      1|   4.0|964982703|
|     1|      3|   4.0|964981247|
|     1|      6|   4.0|964982224|
|     1|      47|   5.0|964983815|
|     1|      50|   5.0|964982931|
|     1|      70|   3.0|964982400|
|     1|     101|   5.0|964980868|
|     1|     110|   4.0|964982176|
|     1|     151|   5.0|964984041|
|     1|     157|   5.0|964984100|
|     1|     163|   5.0|964983650|
|     1|     216|   5.0|964981208|
|     1|    223|   3.0|964980985|
|     1|    231|   5.0|964981179|
|     1|    235|   4.0|964980908|
|     1|    260|   5.0|964981680|
|     1|    296|   3.0|964982967|
|     1|    316|   3.0|964982310|
|     1|    333|   5.0|964981179|
|     1|    349|   4.0|964982563|
+------+-------+------+---------+
only showing top 20 rows

df = ratings.withColumn('userId', ratings['userId'].cast('int')).\
withColumn('movieId', ratings['movieId'].cast('int')).withColumn('rating',
ratings['rating'].cast('float'))
df.printSchema()

root
 |-- userId: integer (nullable = true)
 |-- movieId: integer (nullable = true)
 |-- rating: float (nullable = true)

train, validation, test = df.randomSplit([0.6,0.2,0.2], seed = 0)
print("The number of ratings in each set: {}, {}, {}".format(train.count(), validation.count(), test.count()))

The number of ratings in each set: 60435, 20052, 20349

from pyspark.sql.functions import col, sqrt
def RMSE(predictions):
    squared_diff = predictions.withColumn("squared_diff", 
    pow(col("rating") - col("prediction"), 2))
    mse = squared_diff.selectExpr("mean(squared_diff) as mse").first().mse
    return mse ** 0.5

from pyspark.ml.recommendation import ALS
def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None
    # run Grid Search for all the parameter defined in the range in a loop
    for n in n_factors:
        for reg in reg_param:
            als = ALS(rank = n, 
                      maxIter = num_iterations,
        seed = 0,
        regParam = reg,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop")
      model = als.fit(train)
      predictions = model.transform(valid)
      rmse = RMSE(predictions)
      print('{} latent factors and regularization = {}: validation RMSE is {}'.format(n, reg, rmse))
      # track the best model using RMSE
      if rmse < min_rmse:
          min_rmse = rmse
          best_n = n
          best_reg = reg
          best_model = model

  pred = best_model.transform(train)
  train_rmse = RMSE(pred)
  # best model and its metrics
  print('\nThe best model has {} latent factors and regularization = {}:'.format(best_n, best_reg))
  print('training RMSE is {}; validation RMSE is {}'.format(train_rmse, min_rmse))
  return best_model

from pyspark.sql.functions import col, sqrt
num_iterations = 10
ranks = [6, 8, 10, 12]
reg_params = [0.05, 0.1, 0

Here’s the plain text extracted directly from the image:

```
8 latent factors and regularization = 0.1: validation RMSE is 
0.9168968729472543 
8 latent factors and regularization = 0.2: validation RMSE is 
0.8984989562331739 
8 latent factors and regularization = 0.4: validation RMSE is 
0.9702570878824905 
8 latent factors and regularization = 0.8: validation RMSE is 
1.1934001733725708 
10 latent factors and regularization = 0.05: validation RMSE is 
0.9978579823667801 
10 latent factors and regularization = 0.1: validation RMSE is 
0.917667216467061 
10 latent factors and regularization = 0.2: validation RMSE is 
0.8987281158564604 
10 latent factors and regularization = 0.4: validation RMSE is 
0.9695217416380556 
10 latent factors and regularization = 0.8: validation RMSE is 
1.1934037215306557 
12 latent factors and regularization = 0.05: validation RMSE is 
1.0053856094143756 
12 latent factors and regularization = 0.1: validation RMSE is 
0.9177483804665124 
12 latent factors and regularization = 0.2: validation RMSE is 
0.9000614067181035 
12 latent factors and regularization = 0.4: validation RMSE is 
0.9701108563221951 
12 latent factors and regularization = 0.8: validation RMSE is 
1.1934007238914266 

The best model has 6 latent factors and regularization = 0.2: 
training RMSE is 0.687611382955255; validation RMSE is 
0.8951553355978907 
Total Runtime: 185.58 seconds 

pred_test = final_model.transform(test) 
print('The testing RMSE is ' + str(RMSE(pred_test))) 
The testing RMSE is 0.8959197533497142 
single_user = test.filter(test['userId']==25).select(['movieId','userId']) 
single_user.show() 
+-------+------+ 
|movieId|userId| 
+-------+------+ 
|  7153|    25| 
| 58559|    25| 
| 68954|    25| 
| 91529|    25| 
+-------+------+

single_user.join(movies, single_user.movieId == movies.movieId, 'inner').show()

+-------+------+-------+-------------------------+-------------------------+
|movieId|userId|movieId|title                    |genres                   |
+-------+------+-------+-------------------------+-------------------------+
|  7153 |   25 |  7153 | Lord of the Rings...    | Action|Adventure|...     |
| 58559 |   25 | 58559 | Dark Knight, The...     | Action|Crime|Drama|...    |
| 68954 |   25 | 68954 | Up (2009)               | Adventure|Animation|...   |
| 91529 |   25 | 91529 | Dark Knight Rises...    | Action|Adventure|...      |
+-------+------+-------+-------------------------+-------------------------+

reccomendations = final_model.transform(single_user)
reccomendations.orderBy('prediction', ascending=False).show()

+-------+------+----------+
|movieId|userId|prediction|
+-------+------+----------+
| 68954 |   25 | 4.747929 |
|  7153 |   25 | 4.730045 |
| 58559 |   25 | 4.700764 |
| 91529 |   25 | 4.676375 |
+-------+------+----------+

from pyspark.sql.functions import col, lit

# select a single user from the test set
user_id = 25
single_user_ratings = test.filter(test['userId'] == user_id).select(['movieId', 'userId', 'rating'])

# display the movies the user has liked
print("Movies liked by user with ID", user_id)
single_user_ratings.join(movies, 'movieId').select('movieId', 'title', 'rating').show()

# generate recommendations for the user
all_movies = df.select('movieId').distinct()
user_movies = single_user_ratings.select('movieId').distinct()
movies_to_recommend = all_movies.subtract(user_movies)

# predict ratings for movies the user has not rated yet
recommendations = final_model.transform(movies_to_recommend.withColumn('userId', lit(user_id)))

# filter out the movies that the user has already rated or seen (this
# filters out the movies that the user has not liked as well)
recommendations = recommendations.filter(col('prediction') > 0)

# display the recommendations with movie names
print("Recommended movies for user with ID", user_id)
recommended_movies = recommendations.join(movies,
'movieId').select('movieId', 'title', 'prediction')

# Sort recommended movies by prediction in descending order
ordered_recommendations =
recommended_movies.orderBy(col('prediction').desc())

# Display the ordered recommendations
ordered_recommendations.show()

Movies liked by user with ID 25
+-------+-----------------------+------+
|movieId|title                 |rating|
+-------+-----------------------+------+
|  7153 |Lord of the Rings...  |  5.0 |
| 58559 |Dark Knight, The ...  |  5.0 |
| 68954 |Up (2009)             |  5.0 |
| 91529 |Dark Knight Rises...  |  5.0 |
+-------+-----------------------+------+

Recommended movies for user with ID 25
+-------+-----------------------+----------+
|movieId|title                 |prediction|
+-------+-----------------------+----------+
|  3379 |On the Beach (1959)   |  6.00264 |
| 33649 |Saving Face (2004)    |  5.8678317|
|  7121 |Adam's Rib (1949)     |  5.797302 |
|  5490 |The Big Bus (1976)    |  5.7010717|
|132333 |Seve (2014)           |  5.7010717|
|  6201 |Lady Jane (1986)      |  5.6527076|
|  4495 |Crossing Delancey...  |  5.6527076|
| 60943 |Frozen River (2008)   |  5.648616 |
| 67618 |Strictly Sexual (...) |  5.638301 |
| 25906 |Mr. Skeffington (...) |  5.632532 |
| 77846 |12 Angry Men (1997)   |  5.632532 |
| 84273 |Zeitgeist: Moving...  |  5.626675 |
|138966 |Nasu: Summer in A...  |  5.626675 |
| 26928 |Summer's Tale, A...   |  5.626675 |
|  3819 |Tampopo (1985)        |  5.626675 |
|184245 |De platte jungle ...  |  5.626675 |
| 74226 |Dream of Light (a...  |  5.626675 |
| 26073 |Human Condition I...  |  5.626675 |
|  3200 |Last Detail, The ...  |  5.625157 |
| 27156 |Neon Genesis Evan...  |  5.562292 |
+-------+-----------------------+----------+
