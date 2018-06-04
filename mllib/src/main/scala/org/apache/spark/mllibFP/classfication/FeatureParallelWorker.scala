package org.apache.spark.mllibFP

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllibFP.util.IndexedDataPoint

class SGDWorker(featureDim: Int, workerId: Int){

  var partitionedModel: Array[Double] = Array.fill(featureDim)(0.0)
  def computeDotProduct(features: Vector): Double = {
    val result: Double = features match {
      case SparseVector(Int, Array[Int], Array[Double]) =>{
    0
    }
      case DenseVector(Array[Double]) =>{
        0
    }

    }
  }

}

object FeatureParallelWorker{

  var sgdWorker: SGDWorker = null
  def getSGDWorker(featureDim: Int, workerId: Int): SGDWorker ={
    if(sgdWorker ==  null) {
      new SGDWorker(featureDim, workerId)
    }
    sgdWorker
  }

}