package com.lx.algos.newml.norm

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.{abs, signum}
import com.lx.algos.newml.autograd.GradFunction

/**
  *
  * @project scalaML
  * @author lx on 6:38 PM 11/01/2018
  */

trait NormFunction extends GradFunction

object L1Norm extends NormFunction {
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): DenseMatrix[Double] = signum(theta)

  override def loss(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): Double = sum(abs(theta))
}

object L2Norm extends NormFunction {
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): DenseMatrix[Double] = theta

  override def loss(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): Double = sum(theta.t * theta) / 2.0


}

object DefaultNorm extends NormFunction {
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): DenseMatrix[Double] = DenseMatrix.zeros[Double](theta.rows, theta.cols)

  override def loss(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null, y: DenseMatrix[Double] = null): Double = 0
}