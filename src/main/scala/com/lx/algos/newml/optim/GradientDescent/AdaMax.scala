package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, max}
import breeze.numerics.{abs, pow}
import com.lx.algos.newml.autograd.AutoGrad

/**
  *
  * @project scalaML
  * @author lx on 6:35 PM 24/01/2018
  */

class AdaMax(var lr: Double = 0.002,
             var beta: (Double, Double) = (0.9, 0.999),
             var eps: Double = 1e-8
            ) extends GDOptimizer {

  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //梯度累加信息
    var grad1 = variables.getParam[DenseMatrix[Double]]("grad1", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    //梯度平方累加信息
    var grad2 = variables.getParam[DenseMatrix[Double]]("grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))

    grad1 = beta._1 * grad1 + (1 - beta._1) * grad
    grad2 = max(beta._2 * grad2, abs(grad))

    val bias_correction = 1 - pow(beta._1, epoch)

    val clr = lr / bias_correction
    theta -= clr * grad1 / (grad2 + eps)

    variables.setParam("grad1", grad1)
    variables.setParam("grad2", grad2)
    variables.setParam("theta", theta)
  }

}
