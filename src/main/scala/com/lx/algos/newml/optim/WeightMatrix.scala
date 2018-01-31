package com.lx.algos.newml.optim

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  *
  * @project scalaML
  * @author lx on 7:03 PM 11/01/2018
  */

trait WeightMatrix {
  protected var _theta: DenseMatrix[Double] = null

  protected var fit_intercept = true //是否求偏置

  def weight_init(n_features: Int, n_classes: Int): Unit = {
    if (_theta == null) {
      _theta = DenseMatrix.ones[Double](n_features, n_classes)

    }
  }

  def weight: DenseMatrix[Double] = {

    _theta match {
      case x if (x == null || fit_intercept == false) => _theta
      case _ => _theta(1 until _theta.rows, ::)
    }
  }

  def intercept: DenseVector[Double] = {
    _theta match {
      case x if (x == null) => null
      case y if (fit_intercept) => _theta(0, ::).t
      case _ => DenseVector.zeros(_theta.cols)
    }
  }

}
