from __future__ import print_function

import sys
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext()
    lines = sc.textFile("hdfs://masterNode:9000/user/spark/wordcount/3wishes.txt").cache()

    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)

    # output = counts.collect()
    # for (word, count) in output:
    #     print("%s: %i" % (word, count))
    counts.saveAsTextFile("hdfs://masterNode:9000/user/spark/output/wordcount")
    sc.stop()
