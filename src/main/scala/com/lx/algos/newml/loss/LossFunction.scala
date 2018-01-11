package com.lx.algos.newml.loss

import breeze.linalg.DenseMatrix
import com.lx.algos.newml.autograd.GradFunction

/**
  *
  * @project scalaML
  * @author lx on 4:32 PM 11/01/2018
  */

trait LossFunction extends GradFunction{

  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double], y: DenseMatrix[Double] = null): DenseMatrix[Double] = x

  override def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = x * theta

}
