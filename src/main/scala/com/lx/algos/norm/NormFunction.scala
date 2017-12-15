package com.lx.algos.norm

import breeze.linalg.{DenseMatrix, Matrix}
import com.lx.algos.utils.BaseGradFunction

/**
  *
  * @project scalaML
  * @author lx on 6:50 PM 14/12/2017
  */

trait NormFunction extends BaseGradFunction {

  def grad(theta: Double): Double

  def value(theta: Double): Double

  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null): DenseMatrix[Double] = theta.map(grad)

  override def value(theta: DenseMatrix[Double], x: DenseMatrix[Double] = null): DenseMatrix[Double] = theta.map(value)
}


object L1NormFunction extends NormFunction {
  override def grad(theta: Double): Double = Math.signum(theta)

  override def value(theta: Double) = Math.abs(theta)
}

object L2NormFunction extends NormFunction {
  override def grad(theta: Double): Double = theta

  override def value(theta: Double): Double = theta * theta / 2
}

object DefaultNormFunction extends NormFunction {
  override def grad(theta: Double): Double = 0

  override def value(theta: Double): Double = 0
}