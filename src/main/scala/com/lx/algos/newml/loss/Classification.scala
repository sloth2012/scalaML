package com.lx.algos.newml.loss
import breeze.linalg.{Axis, DenseMatrix, argmax, sum}
import breeze.numerics.exp

/**
  *
  * @project scalaML
  * @author lx on 5:18 PM 11/01/2018
  */

trait ClassificationLoss extends LossFunction {
  def format_value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = { //输出整型值，非概率值，但仍然是onehot形式
    val proba = value(theta, x)
    val res = DenseMatrix.zeros[Double](proba.rows, proba.cols)
    val index = argmax(value(theta, x), Axis._1)

    res(::, index.data.toSeq) := 1.0

    res
  }

}

class SoftmaxLoss extends ClassificationLoss {
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.t * sum(y - value(theta, x)) / (-1.0 * x.rows)
  }

  override def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pred: DenseMatrix[Double] = exp(x * theta)

    //TODO this will be an error
    pred / sum(pred, Axis._1).t
  }
}
