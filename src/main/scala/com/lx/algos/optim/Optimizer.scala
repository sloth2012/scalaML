package com.lx.algos.optim

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, norm}

/**
  *
  * @project scalaML
  * @author lx on 1:53 PM 20/11/2017
  */


trait Optimizer extends WeightVector{

  //迭代终止条件
  var MIN_LOSS: Double = 1e-9

  def input(X: Matrix[Double]): Matrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X.toDenseMatrix)

  def fit(X: Matrix[Double], y: Seq[Double]): Optimizer

  def predict(X: Matrix[Double]): Seq[Double]

  def predict_proba(X: Matrix[Double]): Seq[Double]

  //现在为二分类
  def predict_lr(X: Matrix[Double]): Seq[Double] = {
    assert(weight != null)
    (X * weight + intercept).toArray.map {
      case y: Double => if (y > 0) 1.0 else 0
    }
  }

  def predict_proba_lr(X: Matrix[Double]): Seq[Double] = {
    assert(weight != null)

    (X * weight + intercept).toArray.map {
      case y: Double => 1.0 / Math.log(1 + Math.exp(-y))
    }
  }

  protected def log_print(epoch: Int, acc: Double, avg_loss: Double): Unit = {
    println(s"iteration $epoch: norm:${norm(weight)}, bias:$intercept, avg_loss:$avg_loss, acc:${acc.formatted("%.6f")}")
  }
}
