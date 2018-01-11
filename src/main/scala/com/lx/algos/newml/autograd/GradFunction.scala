package com.lx.algos.newml.autograd

import breeze.linalg.Matrix

/**
  *
  * @project scalaML
  * @author lx on 4:35 PM 11/01/2018
  */

trait GradFunction {
  def grad(theta: Matrix[Double], x: Matrix[Double], y: Matrix[Double] = null): Matrix[Double]

  def value(theta: Matrix[Double], x: Matrix[Double]): Matrix[Double]
}
