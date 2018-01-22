package com.lx.algos.newml.autograd

import breeze.linalg.DenseMatrix

/**
  *
  * @project scalaML
  * @author lx on 4:35 PM 11/01/2018
  */

trait GradFunction {
  def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double] = null): DenseMatrix[Double]

  def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double]
}