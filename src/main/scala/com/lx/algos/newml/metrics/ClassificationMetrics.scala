package com.lx.algos.newml.metrics

import breeze.linalg.{DenseMatrix, Matrix}
import breeze.numerics.log

/**
  *
  * @project scalaML
  * @author lx on 6:32 PM 11/01/2018
  */


object ClassificationMetrics {
  def log_loss(p: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = -y*log(p)
}
