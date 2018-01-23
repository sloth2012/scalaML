package com.lx.algos.newml.autograd

import breeze.linalg.DenseMatrix

/**
  *
  * @project scalaML
  * @author lx on 4:35 PM 11/01/2018
  */

trait GradFunction {
  //求导
  def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): DenseMatrix[Double]

  //loss值
  def loss(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): Double
}