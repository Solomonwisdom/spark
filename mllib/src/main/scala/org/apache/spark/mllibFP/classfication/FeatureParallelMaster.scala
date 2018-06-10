package org.apache.spark.mllibFP.classfication

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllibFP.util.{MLUtils, IndexedDataPoint}

/**
  * @param inputRDD: dataRDD, each element is a IndexedDataPoint
  * @param labels: labels for all datasets
  * @param numFeatures: total number of features
  * @param numPartitions: number of partitions for the model, e.g., number of tasks per stage
  * @param regParam: regularization term
  * @param stepSize: step size for batch
  * @param numIterations
  * @param miniBatchSize
  */
class FeatureParallelMaster(inputRDD: RDD[IndexedDataPoint],
                            labels: Array[Double],
                            numFeatures: Int,
                            numPartitions: Int,
                            regParam: Double,
                            stepSize: Double,
                            numIterations: Int,
                            miniBatchSize: Int) extends Serializable{

  val rand = scala.util.Random
  val sampleIds: Array[Int] = Array.fill(miniBatchSize)(0)
  val coefficients: Array[Double] = Array.fill(miniBatchSize)(0.0)
  val workerFeatureNum = numFeatures / numPartitions +  1  // the last one is different from the before ones
  var lastWorkerFeatureNum = numFeatures - workerFeatureNum * (numPartitions - 1)

  def miniBatchLBFGS(modelName: String): Unit ={

  }


  def miniBatchSGD(modelName: String): Unit = {

    var iter_id: Int = 1

    while(iter_id < numIterations){
      // generate sample ids for mini-batch [sorted in ascending order]
      generateSampleIDArray(sampleIds, labels.length) // may contain duplicate elements

      // 1. broadcast sample ids
      val bcSampleIds: Broadcast[Array[Int]] = inputRDD.sparkContext.broadcast(sampleIds)

      // 2. compute dot products using different workers
      // 3. collect dot products
      val dot_products: Array[Double] = inputRDD.mapPartitions(iter => batchDotProduct(iter, bcSampleIds))
        .reduce(sumArray)

      // 4. compute coefficients for each data point, also evaluate batch loss
      var index_id_batch = 0
      var index_id_global = 0
      val loss: Double = modelName match {
        case "SVM" => {
          var batch_loss: Double = 0
          while (index_id_batch < sampleIds.length) {
            index_id_global = sampleIds(index_id_batch)
            val label_scaled = 2 * labels(index_id_global) - 1
            if ((label_scaled * dot_products(index_id_batch)) < 1) {
              coefficients(index_id_batch) = 0.0 - label_scaled

              batch_loss += 1 - label_scaled * dot_products(index_id_batch) // max(0, 1-2ywx)
            }
            else {
              coefficients(index_id_batch) = 0.0
            }

            index_id_batch += 1
          }
          batch_loss / sampleIds.length // need to consider L2 norm, also incur a all reduce, to fix.
        }
        case "LR" => {
          var batch_loss: Double = 0
          while(index_id_batch < sampleIds.length){
            index_id_global = sampleIds(index_id_batch)
            coefficients(index_id_batch) =
              (1.0 / (1.0 + math.exp(-dot_products(index_id_batch)))) - labels(index_id_global)
            index_id_batch += 1

            val margin = -1.0 * dot_products(index_id_batch)
            if (labels(index_id_global) > 0) {
              // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
              batch_loss += MLUtils.log1pExp(margin)
            } else {
              batch_loss += MLUtils.log1pExp(margin) - margin
            }
          }

          batch_loss / sampleIds.length
        }
        case _ => Double.MaxValue

      }

      println(s"ghand=batchloss: ${loss}")
      // 5. broadcast coefficients
      val bcCoefficients = inputRDD.sparkContext.broadcast(coefficients)
      // 6. update model, include regularization, here we assume it's L2. But l1 is also enabled.
      inputRDD.mapPartitions(
        iter => {
          updateModel(iter, bcCoefficients, bcSampleIds)
          null
        }
      )
      iter_id += 1

    }

  }

  /**
    * element-wise array addition, the results are stored in the first array
    * @param array1
    * @param array2
    * @return
    */
  def sumArray(array1: Array[Double], array2: Array[Double]): Array[Double] = {
    assert(array1.length == array2.length)
    var k: Int = 0
    while(k < array1.length) {
      array1(k) += array2(k)
      k += 1
    }

    array1
  }


  /**
    *
    * @param iter: iterator of dataRDD
    * @param bcCofficients: coefficients of the datapoints, corresponding to the broadcasted data points
    * @param bcSampleIds: ids of sampled data points
    * @return null
    */
  def updateModel(iter: Iterator[IndexedDataPoint], bcCofficients: Broadcast[Array[Double]],
                  bcSampleIds: Broadcast[Array[Int]]): Unit ={
    // use the coefficients to update the model, return dummy results

    val workerId = TaskContext.getPartitionId()
    val featureStartId: Int = workerFeatureNum * workerId
    var localFeatureNum: Int = workerFeatureNum
    if(workerId == numPartitions - 1){
      localFeatureNum = lastWorkerFeatureNum
    }

    val localSgdWorker = FeatureParallelWorker.getSGDWorker(localFeatureNum, featureStartId)

    val localSampleIds: Array[Int] = bcSampleIds.value
    val localCoefficients: Array[Double] = bcCofficients.value
    // dataId, partitionId, Vector
    var sampleCnt = 0

    // update L2 first, because we do in-place update for
    localSgdWorker.updateL2Regu(regParam)

    // has to scan all the data points, but this can be optimized like bucked-based storage
    while(iter.hasNext){
      val dataPoint: IndexedDataPoint = iter.next()
      if(dataPoint.dataId == localSampleIds(sampleCnt)){
        val custom_stepsize = stepSize / miniBatchSize * localCoefficients(sampleCnt)
        localSgdWorker.updateModel(dataPoint.features, custom_stepsize )
        sampleCnt += 1
        while(dataPoint.dataId == localSampleIds(sampleCnt)){
          // each batch may contain duplicate elements, but they are sorted in ascending order
          localSgdWorker.updateModel(dataPoint.features, custom_stepsize )
          sampleCnt += 1
        }
      }

    }

  }


  /**
    * @param iter Iterator for dataRDD
    * @param bcSampleIds sampled ids for the data points
    * @return partial dot products for each sampled data points.
    */
  def batchDotProduct(iter: Iterator[IndexedDataPoint], bcSampleIds: Broadcast[Array[Int]]): Iterator[Array[Double]] = {
    val workerId = TaskContext.getPartitionId()
    val featureStartId: Int = workerFeatureNum * workerId
    var localFeatureNum: Int = workerFeatureNum
    if(workerId == numPartitions - 1){
      localFeatureNum = lastWorkerFeatureNum
    }

    val localSgdWorker = FeatureParallelWorker.getSGDWorker(localFeatureNum, featureStartId)

    val localSampleIds: Array[Int] = bcSampleIds.value
    // dataId, partitionId, Vector
    var sampleCnt = 0
    val localDotProduct: Array[Double] = Array.fill(localSampleIds.length)(0.0)

    // has to scan all the data points, but this can be optimized like bucked-based storage
    while(iter.hasNext){
      val dataPoint: IndexedDataPoint = iter.next()
      if(dataPoint.dataId == localSampleIds(sampleCnt)){
        localDotProduct(sampleCnt) = localSgdWorker.computeDotProduct(dataPoint.features)
        sampleCnt += 1
        while(dataPoint.dataId == localSampleIds(sampleCnt)){
          // each batch may contain duplicate elements, but they are sorted in ascending order
          localDotProduct(sampleCnt) = localDotProduct(sampleCnt - 1)
          sampleCnt += 1
        }
      }

    }

    Iterator(localDotProduct)
  }



  /**
    * generate a sorted Array
    * @param arr
    * @param range
    */
  def generateSampleIDArray(arr: Array[Int], range: Int): Unit ={
    var idx = 0
    while(idx < arr.length){
      arr(idx) = rand.nextInt(range) // [0, range-1]
      idx += 1
    }
    scala.util.Sorting.quickSort(arr)
  }

}


object FeatureParallelMaster{

  // for each method, we define model there.
  def trainMiniBatchSGD(input: RDD[IndexedDataPoint],
                        labels: Array[Double],
                        numFeatures: Int,
                        numPartitions: Int,
                        regParam: Double,
                        stepSize: Double,
                        numIterations: Int,
                        miniBatchSize: Int,
                        modelName: String): Unit ={
    new FeatureParallelMaster(input, labels, numFeatures, numPartitions, regParam, stepSize, numIterations, miniBatchSize)
      .miniBatchSGD(modelName)
  }

  def trainMiniBatchLBFGS(input: RDD[IndexedDataPoint],
                          labels: Array[Double],
                          numFeatures: Int,
                          numPartitions: Int,
                          regParam: Double,
                          stepSize: Double,
                          numIterations: Int,
                          miniBatchSize: Int,
                          modelName: String): Unit ={
    new FeatureParallelMaster(input, labels, numFeatures, numPartitions, regParam, stepSize, numIterations, miniBatchSize)
      .miniBatchLBFGS(modelName)

  }

}