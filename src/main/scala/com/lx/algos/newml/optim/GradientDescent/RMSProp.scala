package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import breeze.numerics.sqrt
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.optim.Optimizer

/**
  *
  * @project scalaML
  * @author lx on 6:06 PM 24/01/2018
  */

class RMSProp(var lr: Double = 0.001,
              var gamma: Double = 0.9,
              var eps: Double = 1e-8
             ) extends Optimizer{
  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //梯度平方累加信息
    var grad2 = variables.getParam[DenseMatrix[Double]]("grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))

    grad2 = gamma * grad2 + (1 - gamma) * grad *:* grad
    val lr_grad = lr / sqrt(grad2 + eps)
    theta -= lr_grad *:* grad

    variables.setParam("grad2", grad2)
    variables.setParam("theta", theta)
  }
}
