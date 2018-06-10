package org.apache.spark.mllibFP.classfication

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

class SGDWorker(featureDim: Int, featureStartId: Int){

  var partitionedModel: Array[Double] = Array.fill(featureDim)(0.0)

  /**
    * compute local dotProduct, since the model is one partition.
    * @param features
    * @return
    */
  def computeDotProduct(features: Vector): Double = {
    val partDotProduct: Double = features match {
      case sp: SparseVector =>{
        // w * x
        var result: Double = 0.0
        var k = 0
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        while(k < index.size){
          // index transform: index - featureStartId
          result += partitionedModel(index(k) - featureStartId) * values(k)
          k += 1
        }
        result
      }
      case dp: DenseVector =>{
        // this is easy, take care of indices
        throw new SparkException("Currently We do not support denseVecor")
      }

    }
    partDotProduct
  }

  /**
    * using one data point to update model, here stepsoze_per_sample is the normalized one by batchsize,
    * also uncludes the coefficients, since gradient is a linear combination of feature vectors.
    * @param features
    * @param stepsize_per_sample
    */
  def updateModel(features: Vector, stepsize_per_sample: Double): Unit ={
    // stepsize_per_sample = stepsize / batchsize * coefficients(i)
    features match {
      case sp: SparseVector => {
        var k = 0
        val index: Array[Int] = sp.indices
        val values: Array[Double] = sp.values
        // sp.size = dimension of the whole vector,
        // index.size = nnz
        while(k < index.size){
          partitionedModel(index(k) - featureStartId) -= stepsize_per_sample * values(k)
          k += 1
        }

      }
      case dp: DenseVector =>{
        throw new SparkException("Currently We do not support denseVecor")
      }
    }
  }

  /**
    * update L2 regularization
    * @param regParam
    */
  def updateL2Regu(regParam: Double): Unit ={
    if(regParam == 0)
      return

    var k = 0
    while(k < partitionedModel.length) {
      partitionedModel(k) *= (1 - regParam)
      k += 1
    }
  }

}

object FeatureParallelWorker{

  var sgdWorker: SGDWorker = null
  def getSGDWorker(featureDim: Int, featureStartId: Int): SGDWorker ={
    if(sgdWorker ==  null) {
      new SGDWorker(featureDim, featureStartId)
    }
    sgdWorker
  }

}