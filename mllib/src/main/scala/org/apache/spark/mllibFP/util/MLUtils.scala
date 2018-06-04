/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllibFP.util

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{MatrixUDT => MLMatrixUDT, VectorUDT => MLVectorUDT}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.{PartitionwiseSampledRDD, RDD}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.execution.datasources.DataSource
import org.apache.spark.sql.execution.datasources.text.TextFileFormat
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.BernoulliCellSampler

/**
  * @param dataId index each data points, the id ranges from 0 to dataNum - 1
  * @param partitionId partitionId of the model, range from 0 to numPartition - 1
  * @param features features can be a SparseVector or DenseVector. Here the feature should be part of the
  *                 whole features.
  */
case class IndexedDataPoint (dataId: Int,
                             partitionId: Int,
                             features: Vector) {
  override def toString: String = {
    s"($dataId, $partitionId, $features)"
  }
}

/**
  * Helper methods to load libsvm dataset and split them into feature parallel.
  * input: a libsvm file, numPartitions
  * output: rdd (index, partOfFeatures) for all datapoints, (labels) to driver (dataId from [0, dataNum-1])
  */
object MLUtils extends Logging {



  /**
    * Loads labeled data in the LIBSVM format into an RDD[IndexedDataPoint].
    * The LIBSVM format is a text-based format used by LIBSVM and LIBLINEAR.
    * Each line represents a labeled sparse feature vector using the following format:
    * {{{label index1:value1 index2:value2 ...}}}
    * where the indices are one-based and in ascending order.
    * This method parses each line into a [[org.apache.spark.mllibFP.util.IndexedDataPoint]],
    * where the feature indices are converted to zero-based.
    *
    * @param sc Spark context
    * @param path file or directory path in any Hadoop-supported file system URI
    * @param numFeatures number of features, which will be determined from the input data if a
    *                    nonpositive value is given. This is useful when the dataset is already split
    *                    into multiple files and you want to load them separately, because some
    *                    features may not present in certain files, which leads to inconsistent
    *                    feature dimensions.
    * @param minPartitions min number of partitions
    * @return labeled data stored as an RDD[LabeledPoint]
    */
  @Since("1.0.0")
  def loadLibSVMFileFeatureParallel(
                      sc: SparkContext,
                      path: String,
                      numFeatures: Int,
                      minPartitions: Int): (RDD[IndexedDataPoint], Array[Int]) = {
    (null, null)
}
//    val parsed = parseLibSVMFile(sc, path, minPartitions)
//
//    // Determine number of features.
//    val d = if (numFeatures > 0) {
//      numFeatures
//    } else {
//      parsed.persist(StorageLevel.MEMORY_ONLY)
//      computeNumFeatures(parsed)
//    }
//
//    parsed.map { case (label, indices, values) =>
//      LabeledPoint(label, Vectors.sparse(d, indices, values))
//    }
//  }

  private[spark] def computeNumFeatures(rdd: RDD[(Double, Array[Int], Array[Double])]): Int = {
    rdd.map { case (label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1
  }

  private[spark] def parseLibSVMFile(
                                      sc: SparkContext,
                                      path: String,
                                      minPartitions: Int): RDD[(Double, Array[Int], Array[Double])] = {
    sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map(parseLibSVMRecord)
  }

  private[spark] def parseLibSVMFile(
                                      sparkSession: SparkSession, paths: Seq[String]): RDD[(Double, Array[Int], Array[Double])] = {
    val lines = sparkSession.baseRelationToDataFrame(
      DataSource.apply(
        sparkSession,
        paths = paths,
        className = classOf[TextFileFormat].getName
      ).resolveRelation(checkFilesExist = false))
      .select("value")

    import lines.sqlContext.implicits._

    lines.select(trim($"value").as("line"))
      .filter(not((length($"line") === 0).or($"line".startsWith("#"))))
      .as[String]
      .rdd
      .map(MLUtils.parseLibSVMRecord)
  }

  private[spark] def parseLibSVMRecord(line: String): (Double, Array[Int], Array[Double]) = {
    val items = line.split(' ')
    val label = items.head.toDouble
    val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
      val indexAndValue = item.split(':')
      val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
    val value = indexAndValue(1).toDouble
      (index, value)
    }.unzip

    // check if indices are one-based and in ascending order
    var previous = -1
    var i = 0
    val indicesLength = indices.length
    while (i < indicesLength) {
      val current = indices(i)
      require(current > previous, s"indices should be one-based and in ascending order;"
        + s""" found current=$current, previous=$previous; line="$line"""")
      previous = current
      i += 1
    }
    (label, indices.toArray, values.toArray)
  }

  /**
    * Loads labeled data in the LIBSVM format into an RDD[LabeledPoint], with the default number of
    * partitions.
    */
  @Since("1.0.0")
  def loadLibSVMFile(
                      sc: SparkContext,
                      path: String,
                      numFeatures: Int): RDD[LabeledPoint] =
    loadLibSVMFile(sc, path, numFeatures, sc.defaultMinPartitions)

  /**
    * Loads binary labeled data in the LIBSVM format into an RDD[LabeledPoint], with number of
    * features determined automatically and the default number of partitions.
    */
  @Since("1.0.0")
  def loadLibSVMFile(sc: SparkContext, path: String): RDD[LabeledPoint] =
    loadLibSVMFile(sc, path, -1)

  /**
    * Returns a new vector with `1.0` (bias) appended to the input vector.
    */
  @Since("1.0.0")
  def appendBias(vector: Vector): Vector = {
    vector match {
      case dv: DenseVector =>
        val inputValues = dv.values
        val inputLength = inputValues.length
        val outputValues = Array.ofDim[Double](inputLength + 1)
        System.arraycopy(inputValues, 0, outputValues, 0, inputLength)
        outputValues(inputLength) = 1.0
        Vectors.dense(outputValues)
      case sv: SparseVector =>
        val inputValues = sv.values
        val inputIndices = sv.indices
        val inputValuesLength = inputValues.length
        val dim = sv.size
        val outputValues = Array.ofDim[Double](inputValuesLength + 1)
        val outputIndices = Array.ofDim[Int](inputValuesLength + 1)
        System.arraycopy(inputValues, 0, outputValues, 0, inputValuesLength)
        System.arraycopy(inputIndices, 0, outputIndices, 0, inputValuesLength)
        outputValues(inputValuesLength) = 1.0
        outputIndices(inputValuesLength) = dim
        Vectors.sparse(dim + 1, outputIndices, outputValues)
      case _ => throw new IllegalArgumentException(s"Do not support vector type ${vector.getClass}")
    }
  }




  /**
    * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
    * overflow. This will happen when `x > 709.78` which is not a very large number.
    * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
    *
    * @param x a floating-point value as input.
    * @return the result of `math.log(1 + math.exp(x))`.
    */
  private[spark] def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  private[mllib] lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }
}
