from __future__ import print_function

import sys
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext()
    lines = sc.textFile("hdfs://localhost:9000/user/sebastian/input/3wishes.txt").cache()

    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)

    # output = counts.collect()
    # for (word, count) in output:
    #     print("%s: %i" % (word, count))
    counts.saveAsTextFile("hdfs://localhost:9000/user/sebastian/output")
    sc.stop()
