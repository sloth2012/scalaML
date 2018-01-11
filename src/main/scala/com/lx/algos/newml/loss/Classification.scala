package com.lx.algos.newml.loss
import breeze.linalg.{Matrix, sum}
import breeze.numerics.{exp, log}

/**
  *
  * @project scalaML
  * @author lx on 5:18 PM 11/01/2018
  */

trait ClassificationLoss extends LossFunction
//TODO y onehotencoding, theta: (n+1)*
class SoftmaxLoss extends ClassificationLoss {
  override def grad(theta: Matrix[Double], x: Matrix[Double], y: Matrix[Double]): Matrix[Double] = {
    x.t * sum(value(theta, x) - y) / x.rows
  }

  override def value(theta: Matrix[Double], x: Matrix[Double]): Matrix[Double] = {
    val pred: Matrix[Double] = exp(x * theta)
    pred / sum(pred)
  }
}
