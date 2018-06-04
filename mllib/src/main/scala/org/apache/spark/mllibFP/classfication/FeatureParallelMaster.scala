package org.apache.spark.mllibFP.util

import org.apache.spark.TaskContext
import org.apache.spark.mllibFP.FeatureParallelWorker
import org.apache.spark.rdd.RDD

class FeatureParallelMaster(inputRDD: RDD[IndexedDataPoint],
                            labels: Array[Int],
                            numFeatures: Int,
                            numPartitions: Int,
                            regParam: Double,
                            stepSize: Double,
                            numIterations: Int,
                            miniBatchSize: Int) extends Serializable{

  val rand = scala.util.Random

  def miniBatchLBFGS(modelName: String): Unit ={

  }


  def miniBatchSGD(modelName: String): Unit = {

    val iter = 1
    val sampleIds: Array[Int] = Array.fill(miniBatchSize)(0)
    val coefficients: Array[Double] = Array.fill(miniBatchSize)(0.0)
    val firstFeatureNum = numFeatures / numPartitions +  1  // the last one is different from the before ones
    var lastWorkerFeatureNum = numFeatures - firstFeatureNum * (numPartitions - 1)

    while(iter < 100){
      // generate sample ids for mini-batch [sorted in ascending order]
      generatesampleIDArray(sampleIds, labels.length)

      // broadcast sample ids
      val bcSampleIds = inputRDD.sparkContext.broadcast(sampleIds)

      // compute dot products using different workers
      // each worker contains numFeatures / numPartitions + 1,
      // the last one contains (numFeatures - (numFeatures / numPartitions + 1) * (numPartitions - 1) features
      val dot_products: Array[Double] = inputRDD.mapPartitions{
        (iter: Iterator[IndexedDataPoint]) => {
          var localFeatureNum = firstFeatureNum
          val workerId = TaskContext.getPartitionId()
          if(workerId == numPartitions - 1){
            localFeatureNum = lastWorkerFeatureNum
          }
          val sgdWorker = FeatureParallelWorker.getSGDWorker(localFeatureNum, workerId)

          val localSampleIds: Array[Int] = bcSampleIds.value
          // dataId, partitionId, Vector
          val sampleCnt = 0
          val dotProduct: Array[Double] = Array.fill(localSampleIds.length)(0.0)
          iter.foreach{
            (indexedDataPoint: IndexedDataPoint) =>{
              if(indexedDataPoint.dataId == localSampleIds(sampleCnt)){

              }
            }
          }

          null
        }
      }.reduce{
        sumArray
      }

      val indexId = 0
      if(SVM){
        while(indexId < sampleIds.length){
          data_id = sampleIds[indexId]
          label_scaled = 2 * labels[data_id] - 1
          if((label_scaled * dot_products[indexId]) < 1){
            coefficients[indexId] = 0.0 - label_scaled
          }
          else{
            coefficients[indexId] = 0.0
          }
          // assume L2 regularization
          coefficients[indexId] += regPara

          indexId += 1
        }
      }
      elif (LR){
        while(indexId < sampleIds.length){
          data_id = sampleIds[indexId]
          coefficients[indexId] =
            (1.0 / (1.0 + math.exp(-dot_products[indexId]))) - labels[data_id]
          indexId += 1
        }


      }
      else{

      }
      iter ++
      val bcCoefficients = sc.broadcast(coefficients)
      updateModel(bcCoefficients)
    }

  }

  def computeDotProduct(dataRDD: RDD[(dataId, Array[Int], Array[Double])], bcSampleIds: Array[Int]){
    val localSampleIds = bcSampleIds.value.copy


  }

  def sumArray(x: Array[Double], y: Array[Double]): Array[Double] = {

  }


  // generate a sorted Array
  def generatesampleIDArray(arr: Array[Int], range: Int): Unit ={
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
                        labels: Array[Int],
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
                          labels: Array[Int],
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