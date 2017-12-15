package com.lx.algos.utils

import breeze.linalg.{DenseMatrix, norm, sum}
import breeze.numerics.pow
import com.lx.algos.loss.LossFunction
import com.lx.algos.norm.NormFunction

/**
  *
  * @project scalaML
  * @author lx on 7:24 PM 14/12/2017
  */


trait BaseGradFunction {
  def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double]

  def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double]
}


class AutoGrad(
                x: DenseMatrix[Double],
                y: DenseMatrix[Double],
                var theta: DenseMatrix[Double],
                lossFunction: LossFunction,
                normFunction: NormFunction,
                lambda: Double = 0.15 //正则化系数
              ) {


  def value: DenseMatrix[Double] = lossFunction.value(theta, x)

  def regular_value: Double = norm(normFunction.value(theta).toDenseVector)

  def grad: DenseMatrix[Double] = (lossFunction.dLoss(value, y).t * lossFunction.grad(theta, x)).reshape(x.cols, 1) + lambda * normFunction.grad(theta, y)

  def avgGrad: DenseMatrix[Double] = grad / (1.0 * x.rows)  reshape(x.cols, 1)

  def loss: DenseMatrix[Double] = lossFunction.loss(value, y) + lambda * regular_value

  def avgLoss: Double = sum(loss) / x.rows

  def updateTheta(newTheta: DenseMatrix[Double]): AutoGrad = {
    theta = newTheta
    this
  }

}