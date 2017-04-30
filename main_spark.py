import utils
import json
import datetime
import numpy
from textblob import TextBlob

from feature import *
# import model
# import config
# import CrossValidator

from pyspark import SparkConf, SparkContext, SQLContext
# from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics


from pyspark.mllib.classification import LogisticRegressionWithSGD, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col

from pyspark.ml.linalg import Vectors as MLVectors
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.regression import LabeledPoint


def main(sc):

    train_id = utils.load("data_id/train.p")
    test_id = utils.load("data_id/test.p")
    test_id = test_id[:5]

    train_id = [[idx] for idx in train_id]
    test_id = [[idx] for idx in test_id]

    # data_f = spark.createDataFrame(train_id, ['biz_id'])
    sqlContext = SQLContext(sc)
    data_f = sqlContext.createDataFrame(test_id, ['biz_id'])

    # Register user defined functions
    # city = udf(lambda b_id: get_city(b_id), StringType())
    state = udf(lambda b_id: get_state(b_id), IntegerType())
    stars = udf(lambda b_id: get_stars(b_id), FloatType())
    popularity = udf(lambda b_id: get_popularity(b_id), IntegerType())
    name_size = udf(lambda b_id: get_name_size(b_id), IntegerType())
    name_polar = udf(lambda b_id: get_name_polar(b_id), FloatType())
    pos_neg_score = udf(lambda b_id: MLVectors.dense(get_PosNeg_score(b_id)), VectorUDT())
    # clarity = udf(lambda b_id: get_clarity(b_id), ArrayType(FloatType()))
    elite_cnt = udf(lambda b_id: get_elite_cnt(b_id), IntegerType())
    label = udf(lambda b_id: get_y(b_id), IntegerType())

    # Generate feature columns
    # data_f = data_f.withColumn("city", city(data_f['biz_id']))
    data_f = data_f.withColumn("state", state(data_f['biz_id']))
    data_f = data_f.withColumn("stars", stars(data_f['biz_id']))
    data_f = data_f.withColumn("popularity", popularity(data_f['biz_id']))
    data_f = data_f.withColumn("name_size", name_size(data_f['biz_id']))
    data_f = data_f.withColumn("name_polar", name_polar(data_f['biz_id']))
    data_f = data_f.withColumn("pos_neg_score", pos_neg_score(data_f['biz_id']))
    # data_f = data_f.withColumn("clarity", clarity(data_f['biz_id']))
    data_f = data_f.withColumn("elite_cnt", elite_cnt(data_f['biz_id']))
    data_f = data_f.withColumn("y", label(data_f['biz_id']))
    data_f.show(5)

    # One-hot encoding
    encoder = OneHotEncoder(inputCol="state", outputCol="stateVec")
    encoded = encoder.transform(data_f)
    encoded.show(5)

    # Assemble columns to features
    assembler = VectorAssembler(
    inputCols=["stateVec","stars","popularity","name_size","name_polar","pos_neg_score","elite_cnt"],
    outputCol="features")

    output = assembler.transform(encoded)
    output.show(5)

    train_d = output
    test_d = output

    train_dd = (train_d.select(col("y"), col("features")) \
                .rdd \
                .map(lambda row: LabeledPoint(row.y, MLLibVectors.fromML(row.features))))
    m = SVMWithSGD.train(train_dd)
    predictionAndLabels = train_d.rdd.map(lambda lp: (float(m.predict(lp.features)), lp.y))
    # Grid search for best params and model
    # scores = {}
    # max_score = 0
    # for m in model_list:
    #     print ('run', m)
    #     evaluator = BinaryClassificationEvaluator()
    #     cv = CrossValidator(estimator=model_list[m],
    #                 estimatorParamMaps=params_list[m],
    #                 evaluator=evaluator,
    #                 numFolds=3)
    #     cv.fit(train)
    #     scores[m] = cv.get_best_score()
    #     if scores[m] > max_score:
    #         op_params = params_list[m][cv.get_best_index()]
    #         op_model = cv.get_best_model()
    #         op_m_name = m

    # predictionAndLabels = test.map(lambda lp: (float(op_model.predict(lp.features)), lp.y))

    # Instantiate metrics object
    bi_metrics = BinaryClassificationMetrics(predictionAndLabels)
    mul_metrics = MulticlassMetrics(predictionAndLabels)

    # Area under precision-recall curve
    print("Area under PR = %s" % bi_metrics.areaUnderPR)
    # Area under ROC curve
    print("Area under ROC = %s" % bi_metrics.areaUnderROC)
    # Confusion Matrix
    print ("Confusion Matrix")
    mul_metrics.confusionMatrix().toArray()
    print ("Accuracy = %s" % mul_metrics.accuracy)
    # print("FP, TP = ", bi_metrics.roc().collect())

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName("Yelp")
    conf = conf.setMaster("local[4]")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)
