package com.lx.algos.newml.loss

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{exp, log}

/**
  *
  * @project scalaML
  * @author lx on 5:18 PM 11/01/2018
  */

trait ClassificationLoss extends LossFunction {
  //theta是包含偏置的矩阵
  def predict(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = { //输出整型值，非概率值，但仍然是onehot形式
    val proba = predict_proba(theta, x)
    val res = DenseMatrix.zeros[Double](proba.rows, proba.cols)
    val index = argmax(proba, Axis._1)

    0 until index.length map { i =>
      res(i, index(i)) = 1.0
    }

    res
  }

  def predict_proba(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double]

  //权重偏置分离的值求解
  def predict(weight: DenseMatrix[Double], intercept: DenseVector[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = { //输出整型值，非概率值，但仍然是onehot形式
    val proba = predict_proba(weight, intercept, x)
    val res = DenseMatrix.zeros[Double](proba.rows, proba.cols)
    val index = argmax(proba, Axis._1)
    0 until index.length map { i =>
      res(i, index(i)) = 1.0
    }

    res
  }

  //权重偏置分离的概率求解
  def predict_proba(weight: DenseMatrix[Double], intercept: DenseVector[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val theta = DenseMatrix.vertcat(intercept.asDenseMatrix.reshape(1, weight.cols), weight)
    val new_x = DenseMatrix.horzcat(DenseMatrix.ones[Double](x.rows, 1), x)
    predict_proba(theta, new_x)
  }

}

class SoftmaxLoss extends ClassificationLoss {

  //output: n*k, n means nfeatures， k means nclasses
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    x.t * (y - predict_proba(theta, x)) / x.rows.toDouble * -1.0
  }

  override def loss(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    -sum(y *:* log(predict_proba(theta, x))) / x.rows.toDouble
  }

  //output: m*k, m means nsamples, k means nclasses
  override def predict_proba(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val pred: DenseMatrix[Double] = exp(x * theta)
    val rowsum = sum(pred, Axis._1)

    0 until pred.rows map {
      case i =>
        pred(i, ::) /= rowsum(i)
    }
    pred
  }

}
