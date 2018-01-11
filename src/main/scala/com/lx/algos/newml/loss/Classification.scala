package com.lx.algos.newml.loss
import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.exp

/**
  *
  * @project scalaML
  * @author lx on 5:18 PM 11/01/2018
  */

trait ClassificationLoss extends LossFunction
//TODO y onehotencoding, theta: (n+1)*
class SoftmaxLoss extends ClassificationLoss {
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.t * sum(y - value(theta, x)) / (-1.0 * x.rows)
  }

  override def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pred: DenseMatrix[Double] = exp(x * theta)
    pred / sum(pred)
  }
}
