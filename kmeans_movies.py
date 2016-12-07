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
from pyspark.mllib.clustering import KMeans, KMeansModel

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
    centroid = np.array([float("{0:.1f}".format(random() * limit)) for i in range(size)])
    return centroid

# Initialize all centroids for the KMeans
def generateInitialCentroids(size, n):
    centroids = [initializeCentroid(size, 5) for i in range(n)]
    return centroids

def createPoint(list, sharedData):
    point = np.zeros(len(sharedData.value))
    for i in range(len(list)):
        movieId, rating = map(float, list[i].split('|'))
        if movieId in sharedData.value:
            point[sharedData.value[movieId]] = rating
    return point

# generate the points for the KMeans
def generatePoints(file, sharedData):
    points = file.map(lambda line: line.split()[1]) \
                .map(lambda line: line.split(',')) \
                .map(lambda line: createPoint(line, sharedData))
    return points

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: kmeans <k> <maxIterations>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="KMeans")
    lines = sc.textFile("hdfs://localhost:9000/user/sebastian/dataset_small/ratings.csv")

    # Formatting data
    ratings = parseDataset(lines)
    sharedData = sc.broadcast(movieIdList(lines))
    size = len(sharedData.value)
    data = generatePoints(ratings, sharedData)

    # KMeans
    start = time()
    k = int(sys.argv[1])
    maxIterations = int(sys.argv[2])
    model = KMeans.train(data, k,
                maxIterations = maxIterations,
                initialModel = KMeansModel(generateInitialCentroids(size, k)))
    end = time()
    elapsed_time = end - start
    output = [
        "Final centers: " + str(model.clusterCenters),
        "Total Cost: " + str(model.computeCost(data)),
        "Value of K: " + str(k),
        "Elapsed time: %0.10f seconds." % elapsed_time
    ]

    info = sc.parallelize(output)
    info.saveAsTextFile("hdfs://localhost:9000/user/sebastian/output")
    sc.stop()
