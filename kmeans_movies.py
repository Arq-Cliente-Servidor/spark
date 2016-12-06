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
from random import random
from time import time
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans

# Format: userId movieId|rating,movieId|rating,movieId|rating...
def parseFile(line):
    myFormat = ""
    flag = False

    for i in range(len(line)):
        myFormat += str(line[i])
        flag = not flag
        if i < len(line) - 1:
            if flag:
                myFormat += "|"
            else:
                myFormat += ","

    return myFormat

def parseDataset(file):
    lines = file.map(lambda line: line.split(',')[:3]) \
                    .map(lambda line: (line[0], line[1:])) \
                    .reduceByKey(lambda value1, value2: value1 + value2) \
                    .sortByKey() \
                    .map(lambda line: str(line[0]) + ' ' + parseFile(line[1]))
    return lines

def movieIdList(file):
    lines = file.map(lambda line: line.split(',')[1]) \
                .collect()

    movieIds = {}
    count = 0

    # Map the films to be able to identify them in the centroids
    for i in range(len(lines)):
        movieId = int(lines[i])
        if not movieId in movieIds:
            movieIds[movieId] = count
            count += 1

    return movieIds

# Create a centroid with random valoes from 0 to 5
def initializeCentroid(size, limit):
    centroid = [float("{0:.1f}".format(random() * limit)) for i in range(size)]
    return centroid

# Initialize all centroids for the KMeans
def generateInitialCentroids(size, n):
    centroids = [initializeCentroid(size, 5) for i in range(n)]
    return centroids

if __name__ == "__main__":
    sc = SparkContext(appName="KMeans")
    ratings = sc.textFile("hdfs://localhost:9000/user/sebastian/dataset_small/ratings.csv")

    data = parseDataset(ratings)
    sharedData = sc.broadcast(movieIdList(ratings))

    sc.stop()
