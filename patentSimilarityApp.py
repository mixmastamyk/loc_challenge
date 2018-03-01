'''

    Simulate real script.

'''
from __future__ import print_function
from datetime import datetime
import re

import numpy as np
import findspark

import pyspark
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2VecModel
from pyspark.sql.functions import udf
from pyspark.ml import PipelineModel
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.feature import Normalizer
from pyspark.sql import SparkSession

import config


findspark.init(config.spark_folder)
spark = df_all = w2v_model = None  # keep global state


def init():
    ''' Initialize libraries. '''
    print('Initializing...')
    sconf = pyspark.SparkConf().setAll([
        ('spark.executor.memory', config.spark_executor_memory),
        ('spark.executor.instances', config.spark_executor_instances),
        ('spark.executor.cores', config.spark_executor_cores),
        #('spark.cores.max', config.spark_cores_max),
        ('spark.driver.memory', config.spark_driver_memory),
        ('master', config.spark_master),
    ])

    global spark, df_all, w2v_model

    spark = SparkSession.builder.appName('similarity2').config(conf=sconf,
                                                              ).getOrCreate()
    spark.sparkContext.setLogLevel(config.spark_log_level)

    df_all = spark.read.parquet(config.input_dir).sample(
                withReplacement=False,
                fraction=config.spark_fraction,
                seed=config.spark_seed
            )
    w2v_model = Word2VecModel.load(config.model_file)


def transform_input(input_text):
    ''' '''
    lines = [(input_text,)]
    df = spark.createDataFrame(lines, ['text'])
    def removePunctuation(text):
        text=text.lower().strip()
        text = re.sub('[^0-9a-zA-Z ]', '', text)
        return text
    remove_punt_udf = udf(removePunctuation, StringType())

    tokenizer = Tokenizer(inputCol='text_noPunct', outputCol='token_text')
    df_new = df.withColumn('text_noPunct', remove_punt_udf('text'))
    df_new = tokenizer.transform(df_new)

    def remove_blank_token(text):
        text = list(filter(lambda x: x!= '', text))
        return text
    remove_blank_token_udf = udf(remove_blank_token, ArrayType(StringType()))
    df_new = df_new.withColumn('token_text',
                               remove_blank_token_udf('token_text'))

    sw_remover = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
    normalizer = Normalizer(inputCol = 'w2v', outputCol = 'w2v_norm')

    pipe = PipelineModel(stages=(sw_remover, w2v_model,
                                 normalizer))
    df_final = pipe.transform(df_new)

    return df_final


def get_similarities(input_text):
    ''' '''
    df_final = transform_input(input_text)
    print('Input Transformed', datetime.time(datetime.now()))

    value = df_final.select('w2v').collect()[0][0]
    def dot_product(vec):
        if (np.linalg.norm(value) * np.linalg.norm(vec)) !=0:
            dot_value = (np.dot(value, vec) / (np.linalg.norm(value) *
                         np.linalg.norm(vec)))
            return dot_value.tolist()
    dot_product_udf = udf(dot_product, FloatType())

    df_all_cos = df_all.withColumn('cos_dis', dot_product_udf('w2v')).dropna(
                                                            subset='cos_dis')
    return df_all_cos


def get_similar_patents(input_text, title=None):
    ''' '''
    df_all_cos = get_similarities(input_text)
    print('Cosine Distances Calculated', datetime.time(datetime.now()))
    top_apps = (df_all_cos.select('appid','inventiontitle','cos_dis')
                                    .orderBy('cos_dis', ascending=False)
                                    .limit(config.query_limit).collect())
    print('DF Sorted', datetime.time(datetime.now()))
    most_similar = []
    bad_apps = config.bad_apps

    for i, row in enumerate(top_apps):
        if not str(top_apps[i][0]) in bad_apps:
            most_similar.append(top_apps[i][0:2])

    return most_similar


#~ def full_pipeline(input_text):
    #~ ''' Function may be unnecessary. '''

    #~ return get_similar_patents(input_text)


if __name__ == '__main__':
    import sys
    from pprint import pprint

    print('results:\n')
    pprint(get_similar_patents(sys.argv[1], title=sys.argv[2]))


