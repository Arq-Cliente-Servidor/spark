    #
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A K-means clustering program using MLlib.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys

import numpy as np
from time import time
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans

def parseVector(line):
    # return np.array([float(x) for x in line.split(' ')])
    return np.array([np.array([float(y) for y in x.split(',')])
                                        for x in line.split('|')])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: kmeans <file> <k>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="KMeans")
    lines = sc.textFile(sys.argv[1])

    data = lines.map(parseVector)
    k = int(sys.argv[2])

    start = time()
    model = KMeans.train(data, k)
    end = time()
    elapsed_time = end - start

    print("Final centers: " + str(model.clusterCenters))
    print("Total Cost: " + str(model.computeCost(data)))
    print("Value of K: " + str(k))
    print("Elapsed time: %0.10f seconds." % elapsed_time)

    sc.stop()
