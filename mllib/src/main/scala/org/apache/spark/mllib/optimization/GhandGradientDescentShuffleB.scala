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

package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors, SparseVector}
import org.apache.spark.rdd.{RDD}
import breeze.linalg.{norm => brzNorm}
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.{HashPartitioner, TaskContext}
import org.apache.spark.broadcast.Broadcast

/**
  * This class use the Breeze library to see whether the linalg used in treeaggregate() is the bottleneck.
  */
/**
  * Class used to solve an optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */

class GhandGradientDescentShuffleB private[spark] (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  //  private var convergenceTol: Double = 0.001
  private var convergenceTol: Double = 0.0
  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is greater than 1, the diff of solution vectors
    *    is compared to relative tolerance which means normalizing by the norm of
    *    the new solution vector.
    *  - If the norm of the new solution vector is less than or equal to 1, the diff of solution
    *    vectors is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
    * :: DeveloperApi ::
    * Runs gradient descent on the given training data.
    *
    * @param data training data
    * @param initialWeights initial weights
    * @return solution vector
    */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GhandGradientDescentShuffleB.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol)
    weights
  }

}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */
@DeveloperApi
object GhandGradientDescentShuffleB extends Logging {
  /**
    * Run stochastic gradient descent (SGD) in parallel using mini batches.
    * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
    * in order to compute a gradient estimate.
    * Sampling, and averaging the subgradients over this subset is performed using one standard
    * spark map-reduce in each iteration.
    *
    * @param data Input data for SGD. RDD of the set of data examples, each of
    *             the form (label, [feature values]).
    * @param gradient Gradient object (used to compute the gradient of the loss function of
    *                 one single data example)
    * @param updater Updater function to actually perform a gradient step in a given direction.
    * @param stepSize initial step size for the first step
    * @param numIterations number of iterations that SGD should be run.
    * @param regParam regularization parameter
    * @param miniBatchFraction fraction of the input data set that should be used for
    *                          one iteration of SGD. Default value 1.0.
    * @param convergenceTol Minibatch iteration will end before numIterations if the relative
    *                       difference between the current weight and the previous weight is less
    *                       than this value. In measuring convergence, L2 norm is calculated.
    *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
    * @return A tuple containing two elements. The first element is a column matrix containing
    *         weights for every feature, and the second element is an array containing the
    *         stochastic loss computed for every iteration.
    */
  def runMiniBatchSGD(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       miniBatchFraction: Double,
                       initialWeights: Vector,
                       convergenceTol: Double): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }
    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference
    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    val weights: DenseVector = Vectors.dense(initialWeights.toArray).toDense
    val num_features = weights.size
    val bcWeights: Broadcast[DenseVector] = data.context.broadcast(weights)
    // first run: broadcast the variable to the cluster.

    // the first model cannot be paralleize from a big Seq, since it maybe to big for the driver.
    val weights_final: DenseVector = repeatMLStart(data.sample(false, miniBatchFraction, 42),
        bcWeights, numIterations, stepSize, num_features, numExamples)

    bcWeights.destroy() // destory the broadcast variable

    if (TaskContext.isDebug) {
      val train_loss = data.map(x => math.max(0, 1.0 - (2.0 * x._1 - 1.0) * dot(x._2, weights_final)))
        .reduce((x, y) => x + y)
      val norm_value_debug = brzNorm(weights_final.asBreeze, 2)
      logInfo(s"ghandTrainLoss=trainLoss:${(train_loss) / numExamples}=" +
        s"weightNorm:${norm_value_debug}")
    }

    (weights_final, stochasticLossHistory.toArray)
  }

  /**
    *
    * @param initModelRDD The initial model RDD[U], where each partition has only one element, i.e.,
    *                     the parameter of the model, which is a breeze vector
    * @param numIter Number of iterations to perform the training
    * @return return model in the DenseVector
    */
  def repeatML(dataRDD: RDD[(Double, Vector)], initModelRDD: RDD[DenseVector],
               numIter: Int, stepSize: Double, numFeatures: Int, numExamples: Long): DenseVector = {
    if (TaskContext.isDebug) {
      val weights_tmp = initModelRDD.take(1)(0)
      val train_loss = dataRDD.map(x => math.max(0, 1.0 - (2.0 * x._1 - 1.0) * dot(x._2, weights_tmp)))
        .reduce((x, y) => x + y)
      val norm_value_debug = brzNorm(weights_tmp.asBreeze, 2)
      logInfo(s"ghandTrainLoss=trainLoss:${(train_loss) / numExamples}=" +
        s"weightNorm:${norm_value_debug}")
      initModelRDD.map{
        dv =>
          brzNorm(dv.asBreeze, 2)
      }.collect().map{
        x => logInfo(s"ghand=SB=weightNorm:${x}")
      }
      // you have to collect, other wise this will be optimized, i.e., it will be not executed.
    }

    if (numIter > 0) {
      // first link the two RDD via the partition Index, then perform SeqOp on each data point
      // and the corresponding model
      val models: RDD[DenseVector] = updateModel(dataRDD, initModelRDD, stepSize)
      // models has numPartitions partitions, each partition with one Iterator,
      // and the Iterator has only one element, U, which is the local model.
      val avergedModels: RDD[DenseVector] = allReduce2(models, numFeatures)
      // here exchange the model, do something like a all-reduce, to communicate the model,
      // the result is return a RDD with numPartitions elements, each element is the parameters
      // of the model
      // BUT: all the elements may not have exactly the same value, it depends on your all-reduce.
      repeatML(dataRDD, avergedModels, numIter - 1, stepSize, numFeatures, numExamples)
    }
    else {
      initModelRDD.take(1)(0)
      // return the first element, which is the model
    }

  }

  def repeatMLStart(dataRDD: RDD[(Double, Vector)], bcWeights: Broadcast[DenseVector],
                    numIter: Int, stepSize: Double, numFeatures: Int, numExamples: Long): DenseVector = {
    if (numIter > 0) {
      // first link the two RDD via the partition Index, then perform SeqOp on each data point
      // and the corresponding model
      val models: RDD[DenseVector] = updateModelStart(dataRDD, bcWeights, stepSize)
      // models has numPartitions partitions, each partition with one Iterator,
      // and the Iterator has only one element, U, which is the local model.
      val avergedModels: RDD[DenseVector] = allReduce2(models, numFeatures)
      // here exchange the model, do something like a all-reduce, to communicate the model,
      // the result is return a RDD with numPartitions elements, each element is the parameters
      // of the model, and all the elements share exactly the same value.
      repeatML(dataRDD, avergedModels, numIter - 1, stepSize, numFeatures, numExamples)
    }
    else {
      bcWeights.value
    }

  }

  /**
    * @param dataRDD
    * @param initModelRDD modelRDD, each partition has exactly one element, which is the full model.
    * @return the newModelRDD
    */
  def updateModel(dataRDD: RDD[(Double, Vector)], initModelRDD: RDD[DenseVector], stepSize: Double): RDD[DenseVector] = {
    //apply data from itt to the model itu. Note that itu only has one element, that is the model.
    val aggregateFunction =
      (itd: Iterator[(Double, Vector)], itm: Iterator[DenseVector]) => {
        val localModel: DenseVector = itm.next()
        val newModel = itd.foldLeft(localModel)(
          (model, dataPoint) => {
            val thisIterStepSize = stepSize
            val dotProduct = dot(dataPoint._2, model)
            val labelScaled = 2 * dataPoint._1 - 1.0
            val local_loss = if (1.0 > labelScaled * dotProduct) {
              axpy((-labelScaled) * (-thisIterStepSize), dataPoint._2, model)
              // What a fucking day! I keeped finding the 'minus' for five hours -- because I am not focused enough.
              1.0 - labelScaled * dotProduct
            } else {
              0
            }
            model
          }
        )
        Iterator(newModel)
    }
    dataRDD.zipPartitions(initModelRDD, preservesPartitioning = false)(aggregateFunction)
  }

  def updateModelStart(dataRDD: RDD[(Double, Vector)], bcWeights: Broadcast[DenseVector],
                       stepSize: Double): RDD[DenseVector] = {
    //apply data from itt to the model itu. Note that itu only has one element, that is the model.
    val mapPartitionsFunc =
      (itd: Iterator[(Double, Vector)]) => {
        // so many transformations, caused by implementation, i.e., polymorphic
        val localModel: DenseVector = bcWeights.value.toDense
        // conform the result to be a iterator of model
        itd.foldLeft(localModel)(
          (model, dataPoint) => {
            val thisIterStepSize = stepSize
            val dotProduct = dot(dataPoint._2, model)
            val labelScaled = 2 * dataPoint._1 - 1.0
            val local_loss = if (1.0 > labelScaled * dotProduct) {
              axpy((-labelScaled) * (-thisIterStepSize), dataPoint._2, model)
              // What a fucking day! I keeped finding the 'minus' for five hours -- because I am not focused enough.
              1.0 - labelScaled * dotProduct
            } else {
              0
            }
            model
          }
       )
    }
    dataRDD.mapPartitions(it => Iterator(mapPartitionsFunc(it)), preservesPartitioning = false)
    // this is not a key value, so whether preserve or not does not make any sense.
  }

  /**
    * calculate the average of the elements inside each partition, and then distribute it back
    * to all the partitions, each partition has exactly one DenseVector
    *
    * @return
    */
  def allReduce(models: RDD[DenseVector], numFeatures: Int): RDD[DenseVector] = {
    // if just no average, it should also work for small iterations like one.
    // each partition has exactly one DenseVector which is the model
    val numPartion = models.getNumPartitions
    val model_with_index = models.mapPartitionsWithIndex {
      (i, iter) =>
        iter.map {
          x =>
            Array((i % numPartion, x), ((i + 1) % numPartion, x))
        }
    }
    model_with_index.flatMap(array => array.iterator)
      .foldByKey(Vectors.zeros(numFeatures).toDense, new HashPartitioner(numPartion)) {
        (x, y) =>
          axpy(1.0, x, y)
          y
      }
      .map{
        x =>
          scal(0.5, x._2)
          x._2
      }
  }

  def allReduce2(models: RDD[DenseVector], numFeatures: Int): RDD[DenseVector] = {
    // implement the real reduce, shuffle 1/numPartitions part of each model, and then shuffle back
    // the total number of communication is 2 * numPartitions * model_size.
    // maybe even twice, since there is an index
    val numPartion = models.getNumPartitions
    logInfo(s"ghand=SB=${numPartion}")
    // 1. transform each partition from one element(a DenseVector, the model) to an Iterator(double) with index
    val doubleRDDModel: RDD[(Int, Double)] = models.mapPartitions(
      dv_iter =>
        dv_iter.map {
          dv => {
            dv.toArray.zipWithIndex.map{
              x => (x._2, x._1)
            }
          }
        }
    ).flatMap(array => array.iterator) // this one is expensive do to GC. like 60s


    // 2. shuffle by key, the number of partition should be the same with the initial modelRDD
    // concate the model, get the RDD[Double], each partition is a collection of double
    val duplicateDoubleModel = doubleRDDModel.foldByKey(0, new HashPartitioner(numPartion)){(x, y) => x + y}
      .map(x => (x._1, x._2 / numPartion.toDouble))
      .mapPartitions(
      //iter[(int, double)] => iter[...]
      iter => iter.map{
        // (int, double) => (key, (int, double))
        x => {
          // duplicate the elements for shuffling
          val array = new Array[(Int, (Int, Double))](numPartion)

          (0 to numPartion - 1).map(i => array(i) = (i, x))
          array
        }
      }
    )
    // here each element is an array, with numPartitions (partitionId, (index, value))s, all the k-v inside a partition are the same.
    // i.e., they are used for shuffling.

    //can use groupByKey.
    val arrayBufferModel = duplicateDoubleModel.flatMap(array => array.iterator)
      .aggregateByKey(new ArrayBuffer[(Int, Double)](0), new HashPartitioner(numPartion))(
        seqOp = {
          //(c, v) => c
          (arrayBuffer, x) => {
            arrayBuffer.append(x)
            arrayBuffer
          }
        },
        combOp = {
          // (c, c) => c
          (arrayBuffer1, arrayBuffer2) =>{
            arrayBuffer1 ++= arrayBuffer2
            // ++= appends a set of numbers to the arraybuffer, += appends only one element
          }

        }
      ).values
    // this one is like 5s.

    val newModels: RDD[DenseVector] = arrayBufferModel.mapPartitions{
      iter => iter.map{
        //arrayBuffer[(int, double)] to denseVector
        arraybuffer => arrayBuffer2DenseVector(arraybuffer)
      }
    }

    // 3. transform it back to the RDD[DenseVector], return it back
    newModels
  }

  def arrayBuffer2DenseVector(arrayBuffer: ArrayBuffer[(Int, Double)]) : DenseVector = {
    val array: Array[(Int, Double)] = arrayBuffer.toArray
    val size: Int = array.size

    val resultArray: Array[Double] = new Array(size)

    var i = 0
    while(i < size){
      resultArray(array(i)._1) = array(i)._2
      i += 1
    }

    Vectors.dense(resultArray).toDense

  }
  /**
    * Alias of `runMiniBatchSGD` with convergenceTol set to default value of 0.001.
    */
  def runMiniBatchSGD(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       miniBatchFraction: Double,
                       initialWeights: Vector): (Vector, Array[Double]) =
    GhandGradientDescentShuffleB.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
      regParam, miniBatchFraction, initialWeights, 0.001)


  private def isConverged(
                           previousWeights: Vector,
                           currentWeights: Vector,
                           convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = brzNorm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(brzNorm(currentBDV), 1.0)
  }

}
