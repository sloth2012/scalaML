package com.lx.algos.newml.loss

import breeze.linalg.Matrix
import com.lx.algos.newml.autograd.GradFunction

/**
  *
  * @project scalaML
  * @author lx on 4:32 PM 11/01/2018
  */

trait LossFunction extends GradFunction{

  override def grad(theta: Matrix[Double], x: Matrix[Double], y: Matrix[Double] = null): Matrix[Double] = x

  override def value(theta: Matrix[Double], x: Matrix[Double]): Matrix[Double] = x * theta

}
