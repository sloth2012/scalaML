package com.lx.algos.newml.norm

import breeze.linalg.Matrix
import breeze.numerics.{abs, signum}
import com.lx.algos.newml.autograd.GradFunction

/**
  *
  * @project scalaML
  * @author lx on 6:38 PM 11/01/2018
  */

trait NormFunction extends GradFunction

object L1Norm extends NormFunction {
  override def grad(theta: Matrix[Double], x: Matrix[Double] = null, y: Matrix[Double] = null): Matrix[Double] = signum(theta)

  override def value(theta: Matrix[Double], x: Matrix[Double] = null): Matrix[Double] = abs(theta)
}

object L2Norm extends NormFunction {
  override def grad(theta: Matrix[Double], x: Matrix[Double] = null, y: Matrix[Double] = null): Matrix[Double] = theta

  override def value(theta: Matrix[Double], x: Matrix[Double] = null): Matrix[Double] = theta * theta / 2
}

object DefaultNorm extends NormFunction {
  override def grad(theta: Matrix[Double], x: Matrix[Double] = null, y: Matrix[Double] = null): Matrix[Double] = Matrix.zeros(theta.rows, theta.cols)

  override def value(theta: Matrix[Double], x: Matrix[Double] = null): Matrix[Double] = Matrix.zeros(theta.rows, theta.cols)
}