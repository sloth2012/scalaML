package com.lx.algos.ml.optim

import java.lang.Math.signum

import breeze.linalg.DenseVector

/**
  *
  * @project scalaML
  * @author lx on 12:18 PM 23/11/2017
  */

class WeightVector {

  protected val MIN_LR_EPS = 1e-6

  protected var _weight: DenseVector[Double] = null

  def weight_init(n: Int): Unit = {
    if(_weight == null) _weight = DenseVector.rand[Double](n + 1)
  }

  def weight: DenseVector[Double] = {
    if(_weight != null) _weight.slice(1, _weight.length)
    else null
  }

  def intercept: Double = {
    if(_weight != null) _weight(0)
    else 0.0
  }
}
