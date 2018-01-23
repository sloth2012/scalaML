package com.lx.algos.newml.autograd

import breeze.linalg.{DenseMatrix, sum}
import com.lx.algos.newml.loss.LossFunction
import com.lx.algos.newml.norm.NormFunction

/**
  *
  * @project scalaML
  * @author lx on 5:26 PM 12/01/2018
  */

//自动求导函数，主要是用于符合原损失函数和正则函数
class AutoGrad(
                x: DenseMatrix[Double],
                y: DenseMatrix[Double],
                var theta: DenseMatrix[Double],
                lossFunction: LossFunction,
                normFunction: NormFunction,
                lambda: Double = 0.15 //正则化系数
              ) {


  def size = x.rows

  def grad: DenseMatrix[Double] = lossFunction.grad(theta, x, y) + lambda * normFunction.grad(theta, x)

  def loss: Double = lossFunction.loss(theta, x, y) + lambda * normFunction.loss(theta)

  def updateTheta(newTheta: DenseMatrix[Double]): AutoGrad = {
    assert(newTheta.cols == theta.cols && newTheta.rows == theta.rows)
    theta = newTheta
    this
  }

}
