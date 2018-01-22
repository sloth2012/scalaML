package com.lx.algos.newml.loss
import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.exp

/**
  *
  * @project scalaML
  * @author lx on 5:18 PM 11/01/2018
  */

trait ClassificationLoss extends LossFunction {
  //theta是包含偏置的矩阵
  def formatted_value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = { //输出整型值，非概率值，但仍然是onehot形式
    val proba = value(theta, x)
    val res = DenseMatrix.zeros[Double](proba.rows, proba.cols)
    val index = argmax(proba, Axis._1)

    res(::, index.data.toSeq) := 1.0
    res
  }

  //权重偏置分离的求解
  def formatted_value(weight: DenseMatrix[Double], intercept: DenseVector[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = { //输出整型值，非概率值，但仍然是onehot形式
    val proba = value(weight, intercept, x)
    val res = DenseMatrix.zeros[Double](proba.rows, proba.cols)
    val index = argmax(proba, Axis._1)

    res(::, index.data.toSeq) := 1.0
    res
  }

  //权重偏置分离的求解
  def value(weight: DenseMatrix[Double], intercept: DenseVector[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val theta = DenseMatrix.vertcat(intercept.asDenseMatrix.reshape(1, weight.cols), weight)
    val new_x = DenseMatrix.horzcat(DenseMatrix.ones[Double](x.rows, 1), x)
    value(theta, new_x)
  }

}

class SoftmaxLoss extends ClassificationLoss {
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.t * (y - value(theta, x)) / (-1.0 * x.rows)
  }

  override def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pred: DenseMatrix[Double] = exp(x * theta)
    val rowsum = sum(pred, Axis._1).t

    0 until pred.rows map {
      case i =>
        pred(i, ::) /= rowsum(i)
    }

    pred
  }


}
