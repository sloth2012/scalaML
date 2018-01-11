package com.lx.algos.newml.metrics

import breeze.linalg.Matrix
import breeze.numerics.log

/**
  *
  * @project scalaML
  * @author lx on 6:32 PM 11/01/2018
  */


object ClassificationMetrics {
  def log_loss(p: Matrix[Double], y: Matrix[Double]): Matrix[Double] = {
    val z: Matrix[Double] = y*log(p)
    -z
  }
}
