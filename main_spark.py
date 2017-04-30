import utils
import json
import datetime
import numpy
from textblob import TextBlob

from feature import *

from pyspark import SparkConf, SparkContext, SQLContext
# from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler


def main(sc):

    train_id = utils.load("data_id/train.p")
    test_id = utils.load("data_id/test.p")
    test_id = test_id[:100]

    train_id = [[idx] for idx in train_id]
    test_id = [[idx] for idx in test_id]

    # data_f = spark.createDataFrame(train_id, ['biz_id'])
    sqlContext = SQLContext(sc)
    data_f = sqlContext.createDataFrame(test_id, ['biz_id'])

    # Register user defined functions
    city = udf(lambda b_id: get_city(b_id), StringType())
    state = udf(lambda b_id: get_state(b_id), StringType())
    stars = udf(lambda b_id: get_stars(b_id), FloatType())
    popularity = udf(lambda b_id: get_popularity(b_id), IntegerType())
    name_size = udf(lambda b_id: get_name_size(b_id), IntegerType())
    name_polar = udf(lambda b_id: get_name_polar(b_id), FloatType())
    pos_neg_score = udf(lambda b_id: get_PosNeg_score(b_id), ArrayType(FloatType()))
    elite_cnt = udf(lambda b_id: get_elite_cnt(b_id), IntegerType())
    label = udf(lambda b_id: get_y(b_id), IntegerType())

    # Generate feature columns
    data_f = data_f.withColumn("city", city(data_f['biz_id']))
    data_f = data_f.withColumn("state", state(data_f['biz_id']))
    data_f = data_f.withColumn("stars", stars(data_f['biz_id']))
    data_f = data_f.withColumn("popularity", popularity(data_f['biz_id']))
    data_f = data_f.withColumn("name_size", name_size(data_f['biz_id']))
    data_f = data_f.withColumn("name_polar", name_polar(data_f['biz_id']))
    data_f = data_f.withColumn("pos_neg_score", pos_neg_score(data_f['biz_id']))
    data_f = data_f.withColumn("elite_cnt", elite_cnt(data_f['biz_id']))
    data_f = data_f.withColumn("y", label(data_f['biz_id']))
    data_f.show(5)

    # One-hot Encoding
    # stringIndexer = StringIndexer(inputCol="city", outputCol="city_Index")
    # indexed = stringIndexer.fit(data_f).transform(data_f)
    # indexed.show(5)
    # stringIndexer = StringIndexer(inputCol="state", outputCol="state_Index")

    # data_f.show(5)


if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName("Yelp")
    conf = conf.setMaster("local[4]")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)
