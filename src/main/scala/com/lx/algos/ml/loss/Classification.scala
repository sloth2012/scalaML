package com.lx.algos.ml.loss

/**
  *
  * @project scalaML
  * @author lx on 2:57 PM 15/11/2017
  */

trait Classification extends LossFunction

class HingeLoss(threshold: Double = 1) extends Classification {
  /** Hinge loss for binary classification tasks with y in {-1,1}
    * *
    * Parameters
    * ----------
    * *
    * threshold : float > 0.0
    * Margin threshold. When threshold=1.0, one gets the loss used by SVM.
    * When threshold=0.0, one gets the loss used by the Perceptron.
    */

  assert(threshold >= 0)

  override def loss(p: Double, y: Double): Double = Math.max(0, threshold - p * y)

  override def dLoss(p: Double, y: Double): Double = {
    val z = p * y
    if (z >= threshold) 0
    else -y
  }
}

class SquaredHingeLoss(threshold: Double = 1) extends Classification {
  /** Squared Hinge loss for binary classification tasks with y in {-1,1}
    * *
    * Parameters
    * ----------
    * *
    * threshold : float > 0.0
    * Margin threshold. When threshold=1.0, one gets the loss used by
    * (quadratically penalized) SVM.
    */
  assert(threshold >= 0)

  override def loss(p: Double, y: Double): Double = {
    val z = threshold - p * y
    Math.max(0, 0.5 * z * z)
  }

  override def dLoss(p: Double, y: Double): Double = {
    val z = threshold - p * y
    if (z > 0) -y * z else 0
  }
}

class ModifiedHuber extends Classification {
  /** Modified Huber loss for binary classification with y in {-1, 1}
    * *
    * This is equivalent to quadratically smoothed SVM with gamma = 2.
    * *
    * See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    * Stochastic Gradient Descent', ICML'04.
    */

  override def loss(p: Double, y: Double): Double = {
    val z = p * y
    if (z >= 1.0) 0.0
    else if (z >= 1.0) (1 - z) * (1 - z)
    else -4 * z
  }

  override def dLoss(p: Double, y: Double): Double = {
    val z = p * y
    if (z >= 1.0) 0
    else if (z >= -1) -2 * (1 - z) * y
    else -4 * y

  }
}

class LogLoss extends Classification {
  /** Logistic regression loss for binary classification with y in {-1, 1}
    * 用于交叉熵和lR，18是对目标求导做了一部分近似优化
    */
  override def loss(p: Double, y: Double): Double = {
    val z = p * y

    if (z > 18) Math.exp(-z)
    else if (z < -18) -z
    else Math.log(1 + Math.exp(-z))
  }

  override def dLoss(p: Double, y: Double): Double = {
    val z = p * y
    if (z > 18) -y * Math.exp(-z)
    else if (z < -18) -y
    else -y / (1 + Math.exp(z))
  }
}
