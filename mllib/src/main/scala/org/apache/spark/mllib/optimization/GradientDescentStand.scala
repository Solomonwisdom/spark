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
import breeze.linalg.{norm, DenseVector => BDV}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.mllib.WhetherDebug
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD

// This class is only used for MLlib with feature scaling.
/**
 * Class used to solve an optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
class GradientDescentStand private[spark] (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
//  private var convergenceTol: Double = 0.001
  private var convergenceTol: Double = 0.0
  private var ghandNumFeatures: Int = -1

  def setGhandNumFeatures(numFeatures: Int): this.type ={
    this.ghandNumFeatures = numFeatures
    this
  }
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
    val (weights, _) = GradientDescentStand.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol,
      ghandNumFeatures)
    weights
  }

}

/**
 * :: DeveloperApi ::
 * Top-level method to run gradient descent.
 */
@DeveloperApi
object GradientDescentStand extends Logging {
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
      convergenceTol: Double,
      ghandNumFeatures: Int): (Vector, Array[Double]) = {

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

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescentStand.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // ghand: add ways to pre-process the data to calculate the variance.
    val instances: RDD[Instance] = data.map(
      x=>
        Instance(x._1, 1, x._2.asML)
    )

    val summarizer = {
      val seqOp = (c: MultivariateOnlineSummarizer,
                   instance: Instance) =>{
        c.add(Vectors.fromML(instance.features), instance.weight)
      }

      val combOp = (c1: MultivariateOnlineSummarizer,
                    c2: MultivariateOnlineSummarizer) =>{
        c1.merge(c2)
      }

      instances.treeAggregate(
        (new MultivariateOnlineSummarizer)
      )(seqOp, combOp, 3)
    }

    val featuresStd: Array[Double] = summarizer.variance.toArray.map(math.sqrt)
//    val featuresStd: Array[Double] = Array.fill(ghandNumFeatures)(1.0)


    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    train_start_time = System.currentTimeMillis()
    while (!converged && i <= numIterations) {

      logInfo(s"ghandCP=IterationId:${i}=BroadcastStartsTime:${System.currentTimeMillis()}")
      val bcWeights = data.context.broadcast(weights)
      logInfo(s"ghandCP=IterationId:${i}=BroadcastEndsTime:${System.currentTimeMillis()}")

      var time_cal_loss: Double = 0.0
      if (WhetherDebug.isDebug) {
        val start_time = System.currentTimeMillis()

        val train_loss = data.map(x => math.max(0, 1.0 - (2.0 * x._1 - 1.0) * dot(x._2, bcWeights.value)))
          .reduce((x, y) => x + y)
        val breeze_weight = bcWeights.value.asBreeze.toDenseVector
        val norm_value_debug = norm(breeze_weight, 2)

        val end_time = System.currentTimeMillis()

        time_cal_loss = (end_time - start_time) / 1000.0
        logInfo(s"ghand=Iteration:${numIterations - i}" +
          s"=TimeCalLoss:${time_cal_loss}")
        logInfo(s"ghandTrainLoss=weightNorm:${norm_value_debug}=" +
          s"trainLoss:${(train_loss) / numExamples + 0.5 * norm_value_debug * norm_value_debug * regParam}")

      }
      train_end_time = System.currentTimeMillis()
      logInfo(s"ghand=Iteration:${numIterations - i}=" +
        s"TimeWithOutLoss:${(train_end_time - train_start_time) / 1000.0 - time_cal_loss}")
      train_start_time = System.currentTimeMillis()

      val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          }
        )
      logInfo(s"ghandCP=IterationId:${i}=DestroyBroadcastStartsTime:${System.currentTimeMillis()}")
      bcWeights.destroy(blocking = false)
      logInfo(s"ghandCP=IterationId:${i}=DestroyBroadcastEndsTime:${System.currentTimeMillis()}")

      if (miniBatchSize > 0) {
        /**
         * lossSum is computed using the weights from the previous iteration
         * and regVal is the regularization value computed in the previous iteration as well.
         */
        stochasticLossHistory += lossSum / miniBatchSize + regVal
        logInfo(s"ghandCP=IterationId:${i}=updateWeightOnDriverStartsTime:${System.currentTimeMillis()}")

        //--------------------------------------------------------------------//
        // here we use the gradient to update the model.
        // We assume the L2 regularization is zero. For L2 that is not zero, we never use feature scaling.
//        val update = updater.compute(
//          weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
//          stepSize, i, regParam)

        // weight = weight - stepsize * gradientsum / miniBatchSize.toDouble / featureStd
        val weights_array: Array[Double] = weights.toArray
        val gradient_array: Array[Double] = gradientSum.toDenseVector.toArray
        val feature_dim = featuresStd.size
        var k = 0
        while(k < feature_dim) {
          if(featuresStd(k) != 0) {
            val tmp_grad = gradient_array(k) / miniBatchSize.toDouble + weights_array(k) * regParam
            weights_array(k) -= stepSize * tmp_grad / (featuresStd(k) * featuresStd(k))
          }
          k += 1
        }
        //--------------------------------------------------------------------//

        weights = Vectors.dense(weights_array)
        regVal = 0

        logInfo(s"ghandCP=IterationId:${i}=updateWeightOnDriverEndsTime:${System.currentTimeMillis()}")

        logInfo(s"ghandCP=IterationId:${i}=JudgeConvergeStartsTime:${System.currentTimeMillis()}")
        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
        logInfo(s"ghandCP=IterationId:${i}=JudgeConvergeEndsTime:${System.currentTimeMillis()}")

      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    logInfo("GradientDescentStand.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)

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
    GradientDescentStand.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
                                    regParam, miniBatchFraction, initialWeights, 0.001, -1)


  private def isConverged(
      previousWeights: Vector,
      currentWeights: Vector,
      convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }
  var train_start_time: Long = 0
  var train_end_time: Long = 0
}
