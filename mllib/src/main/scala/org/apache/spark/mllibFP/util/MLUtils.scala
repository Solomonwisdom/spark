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

import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.{RDD}
import org.apache.spark.storage.StorageLevel

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
    * @param num_features number of features, which will be determined from the input data if a
    *                    nonpositive value is given. This is useful when the dataset is already split
    *                    into multiple files and you want to load them separately, because some
    *                    features may not present in certain files, which leads to inconsistent
    *                    feature dimensions.
    * @param min_partitions min number of partitions
    * @return labeled data stored as an RDD[LabeledPoint]
    */
  @Since("1.0.0")
  def loadLibSVMFileFeatureParallel(
                      sc: SparkContext,
                      path: String,
                      num_features: Int,
                      min_partitions: Int): (RDD[IndexedDataPoint], Array[Double]) = {

    val parsed: RDD[(Int, Double, Array[Int], Array[Double])] = parseLibSVMFile(sc, path, min_partitions)
    // local_data_id, label, indices, values

    parsed.persist(StorageLevel.MEMORY_ONLY)

    // Determine number of features.
    val d = if (num_features > 0) {
      num_features
    } else {
      computeNumFeatures(parsed)
    }

    // compute number of elements in each partition
    val partition_size: Array[(Int, Int)] = parsed.mapPartitionsWithIndex{
      (partitionId, iter) => {
        Iterator((partitionId, iter.size))
      }
    }.collect()
    val num_data_points_per_partition: Array[Int] = new Array[Int](min_partitions)
    var i =0
    while(i < partition_size.length){
      num_data_points_per_partition(partition_size(i)._1) = partition_size(i)._2
      i += 1
    }
    val global_startId_per_partition: Array[Int] = new Array[Int](min_partitions)
    global_startId_per_partition(0) = 0
    i = 1
    while(i < num_data_points_per_partition.length){
      global_startId_per_partition(i) = num_data_points_per_partition(i - 1) + global_startId_per_partition(i - 1)
      i += 1
    }
    logInfo(s"FP:number of data points is: ${global_startId_per_partition(i-1) + num_data_points_per_partition(i - 1)}")

    val bc_global_startId_per_partition: Broadcast[Array[Int]] = sc.broadcast(global_startId_per_partition)

    // compute the labels according to global index.
    val global_labels: Array[(Int, Double)] = parsed.mapPartitions{
      iter => {
        val local_startID: Int = bc_global_startId_per_partition.value(TaskContext.getPartitionId())
        iter.map{
          //  (localId: Int, label: Double, indices: Array[Int]. value: Array[Double])
          data_point => {
            (data_point._1 + local_startID, data_point._2)
          }
        }
      }
    }.collect()
    val labels: Array[Double] = new Array[Double](global_labels.length)
    i = 0
    while(i < labels.length){
      labels(global_labels(i)._1) = global_labels(i)._2
    }

    // split the data points vertically according to the feature num
    // partition_id, global_data_id, indices, values
    val global_data_point: RDD[(Int, Array[Int], Array[Double])] = parsed.mapPartitions{
      iter => {
        val local_startID: Int = bc_global_startId_per_partition.value(TaskContext.getPartitionId())
        iter.map{
          //  (localId: Int, label: Double, indices: Array[Int]. value: Array[Double])
          data_point => {
            (data_point._1 + local_startID, data_point._3, data_point._4)
          }
        }
      }
    }

    // (partitionId, (global_data_id, indices, values))
    val global_splitted_data_point: RDD[(Int, (Int, Array[Int], Array[Double]))] = global_data_point.mapPartitions{
      iter => {
        iter.map(
          data_point => splitDataPoint(data_point, min_partitions, num_features).toIterator
        )
      }.flatMap(x => x)
    }


    // groupByKey, then sort inside each partition, should return correct results.
    // can also store the whole data as an big array, to have O(1) random access.
    // can even implement CSR format, for efficient storage and O(1) random access.
    val indexed_data_point: RDD[(IndexedDataPoint)] = global_splitted_data_point.groupByKey(min_partitions).mapPartitions(
      iter => {
        // one executor contains only one partition
        val partition: (Int, Iterable[(Int, Array[Int], Array[Double])]) = iter.next()
        val partition_id = partition._1
        val part_data_points: Array[(Int, Array[Int], Array[Double])] = partition._2.toArray
        part_data_points.sortWith(sortByDataID)
        part_data_points.map(
          data_point_in_array_format =>{
          val data_id: Int = data_point_in_array_format._1
          val indices: Array[Int] = data_point_in_array_format._2
          val values: Array[Double] = data_point_in_array_format._3
          IndexedDataPoint(data_id, partition_id, new SparseVector(num_features, indices, values))
          // don't use toArray to SparseVector because it is super inefficient.
          }
        ).toIterator
      }
    )
    indexed_data_point.persist(StorageLevel.MEMORY_ONLY)
    parsed.unpersist()

    (indexed_data_point, labels)
  }


  def sortByDataID(a: (Int, Array[Int], Array[Double]), b: (Int, Array[Int], Array[Double])) : Boolean = {
    a._1 > b._1
  }

  private[spark] def computeNumFeatures(rdd: RDD[(Int, Double, Array[Int], Array[Double])]): Int = {
    rdd.map { case (local_data_id, label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1
  }


  /**
    * split one data point into ${min_partitions}, to be distributed over the cluster.
    * @param data_point (globalDataId, indices, values)
    * @param min_partitions num_partitons vertically
    * @param num_features total number of features
    * @return (partitionId, (globalDataId, indices, values))
    */
  def splitDataPoint(data_point: (Int, Array[Int], Array[Double]), min_partitions: Int,
                     num_features: Int): Array[(Int, (Int, Array[Int], Array[Double]))] ={
    val worker_feature_num = num_features / min_partitions + 1
    val last_worker_feature_num = num_features - worker_feature_num * (min_partitions - 1)

    val indices: Array[Int] = data_point._2
    val values: Array[Double] = data_point._3
    val result: Array[(Int, (Int, Array[Int], Array[Double]))] = new Array[(Int, (Int, Array[Int], Array[Double]))](min_partitions)
    var partition_id = 0
    var tmp_s = 0
    var tmp_e = 0
    var slice_start = 0
    var slice_end = 0
    while(partition_id < min_partitions){
      slice_start = partition_id * worker_feature_num //  included
      slice_end = slice_start + worker_feature_num // excluded
      if(slice_end > num_features)
        slice_end = num_features
      while(tmp_e < indices.length && indices(tmp_e) < slice_end){
        tmp_e += 1
      }
      //      print(s"partitionId:${partition_id} slice_start:${slice_start} slice_end:${slice_end} tmp_s:${tmp_s} tmp_e:${tmp_e}\n")

      val new_indices: Array[Int] = indices.slice(tmp_s, tmp_e)
      val new_values: Array[Double] = values.slice(tmp_s, tmp_e)
      result(partition_id) = (partition_id, (data_point._1, new_indices, new_values))

      partition_id += 1
      tmp_s = tmp_e
    }
    result
  }


  private[spark] def parseLibSVMFile(
                                      sc: SparkContext,
                                      path: String,
                                      minPartitions: Int): RDD[(Int, Double, Array[Int], Array[Double])] = {
    sc.textFile(path, minPartitions)
      .mapPartitions{
        var local_dataPoint_id: Int = -1
        iter: Iterator[String] => {
          iter.map(_.trim)
            .filter(line => !(line.isEmpty || line.startsWith("#")))
            .map(
              line => {
                local_dataPoint_id += 1
                parseLibSVMRecord(local_dataPoint_id, line)
              }
            )
        }
      }
  }

  /**
    * @param localDataPointId, the index of the data point in this line.
    * @param line a data point in libsvm format
    * @return
    */
  private[spark] def parseLibSVMRecord(localDataPointId: Int, line: String): (Int, Double, Array[Int], Array[Double]) = {
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
    (localDataPointId, label, indices.toArray, values.toArray)
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

}
