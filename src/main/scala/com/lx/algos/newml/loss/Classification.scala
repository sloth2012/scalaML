package com.lx.algos.newml.loss

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax, max, sum}
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

  //防止归一化时相乘为0
  private val eps = 1e-8

  //output: n*k, n means nfeatures， k means nclasses
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    -x.t * (y - predict_proba(theta, x)) / x.rows.toDouble
  }

  //TODO log值为0
  override def loss(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    -sum(y *:* log(predict_proba(theta, x))) / x.rows.toDouble
  }

  //output: m*k, m means nsamples, k means nclasses
  override def predict_proba(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    var value = x * theta
    val maxV = max(value)

    //若值过大，则进行删减，避免出现infinity；若过小，则进行增加，避免出现NaN
    value -= {
      if(maxV < 1) -1 else maxV
    }

    //加入eps，防止所有预测为0的情况，属于平滑策略
    val pred: DenseMatrix[Double] = exp(value) + eps
    val rowsum = sum(pred, Axis._1)
    0 until pred.rows map {
      case i =>
        pred(i, ::) /= rowsum(i)
    }
    pred
  }

}
