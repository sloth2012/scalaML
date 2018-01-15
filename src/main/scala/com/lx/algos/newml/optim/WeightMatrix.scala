package com.lx.algos.newml.optim

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  *
  * @project scalaML
  * @author lx on 7:03 PM 11/01/2018
  */

trait WeightMatrix {
  protected var _theta: DenseMatrix[Double] = null

  def weight_init(n_features: Int, n_classes: Int): Unit = {
    if(_theta == null) _theta = DenseMatrix.rand[Double](n_features + 1, n_classes)
  }

  def weight: DenseMatrix[Double] = {
    if(_theta != null) _theta(1 until _theta.rows, ::)
    else null
  }

  def intercept: DenseVector[Double] = {
    if(_theta != null) _theta(0, ::).t
    else DenseVector.zeros(_theta.cols)
  }

}
