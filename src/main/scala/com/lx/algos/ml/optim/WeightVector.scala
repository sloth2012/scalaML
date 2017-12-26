package com.lx.algos.ml.optim

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  *
  * @project scalaML
  * @author lx on 12:18 PM 23/11/2017
  */

class WeightVector {

  protected val MIN_LR_EPS = 1e-6

  protected var _theta: DenseMatrix[Double] = null

  def weight_init(n: Int): Unit = {
    if(_theta == null) _theta = DenseMatrix.rand[Double](n + 1, 1)
  }

  def weight: DenseVector[Double] = {
    if(_theta != null) _theta.toDenseVector.slice(1, _theta.rows)
    else null
  }

  def intercept: Double = {
    if(_theta != null) (_theta.toDenseVector)(0)
    else 0.0
  }
}
