package com.lx.algos.newml.metrics

import breeze.linalg.{DenseMatrix, Matrix, sum}
import breeze.numerics.log

/**
  *
  * @project scalaML
  * @author lx on 6:32 PM 11/01/2018
  */


object ClassificationMetrics {
  def log_loss(p: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {

    assert(p.rows == y.rows && p.cols == y.cols)
    sum(-y*log(p))/y.rows
  }

  def accuracy_score(p: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    assert(p.rows == y.rows && p.cols == y.cols)

    0
  }

}
