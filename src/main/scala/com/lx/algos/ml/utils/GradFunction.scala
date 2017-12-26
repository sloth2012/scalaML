package com.lx.algos.ml.utils

import breeze.linalg.{DenseMatrix, DenseVector, norm, sum}
import com.lx.algos.ml.loss.LossFunction
import com.lx.algos.ml.norm.NormFunction

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

  private def regular_value: Double = norm(normFunction.value(theta).toDenseVector)

  def grad: DenseMatrix[Double] = (lossFunction.dLoss(value, y).t * lossFunction.grad(theta, x)).reshape(x.cols, 1) + lambda * normFunction.grad(theta, y)

  def avgGrad: DenseMatrix[Double] = grad / (1.0 * x.rows) reshape(x.cols, 1)

  def loss: DenseMatrix[Double] = lossFunction.loss(value, y) + lambda * regular_value

  def avgLoss: Double = sum(loss) / x.rows

  def updateTheta(newTheta: DenseMatrix[Double]): AutoGrad = {
    assert(newTheta.cols == theta.cols && newTheta.rows == theta.rows)
    theta = newTheta
    this
  }

}

//该类别主要用于GradientDescent包下的单一更新元素使用，未来将考虑重构该模块
class SimpleAutoGrad(x: DenseVector[Double],
                     y: Double,
                     var theta: DenseMatrix[Double],
                     lossFunction: LossFunction,
                     normFunction: NormFunction,
                     lambda: Double = 0.15 //正则化系数
                    ) {
  val sag = new AutoGrad(x.toDenseMatrix.reshape(1, x.length), DenseVector(y).toDenseMatrix.reshape(1, 1), theta, lossFunction, normFunction, lambda)
  def loss: Double = sag.avgLoss
  def grad: DenseVector[Double] = sag.avgGrad.toDenseVector
  def updateTheta(newTheta: DenseMatrix[Double]): SimpleAutoGrad = {
    theta = newTheta
    sag.updateTheta(newTheta.toDenseMatrix.reshape(x.length, 1))
    this
  }
}
