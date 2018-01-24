package com.lx.algos.newml.metrics

import breeze.linalg.{Axis, DenseMatrix, argmax, sum}
import breeze.numerics.log

/**
  *
  * @project scalaML
  * @author lx on 6:32 PM 11/01/2018
  */


object ClassificationMetrics {
  def log_loss(p: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {

    assert(p.rows == y.rows && p.cols == y.cols)
    sum(-y *:* log(p)) / y.rows
  }

  //假设输入的都是onehot之后的结果
  def accuracy_score(p: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    assert(p.rows == y.rows && p.cols == y.cols)

    val p_index = argmax(p, Axis._1)
    val y_index = argmax(y, Axis._1)

    var counter: Double = 0

    0 until p.rows map { i =>
      if (p_index(i) == y_index(i)) counter += 1
    }

    counter / p.rows
  }

}
