package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import breeze.numerics.sqrt
import com.lx.algos.newml.autograd.AutoGrad

/**
  *
  * @project scalaML
  * @author lx on 5:08 PM 24/01/2018
  */

class AdaGrad(var lr: Double = 0.01,
             var eps: Double = 1e-8
             ) extends GDOptimizer {
  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //梯度平方累加信息
    val grad2 = variables.getParam[DenseMatrix[Double]]("grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    grad2 += grad *:* grad
    val lr_grad = lr / sqrt(grad2 + eps)
    theta -= lr_grad *:* grad

    variables.setParam("grad2", grad2)
    variables.setParam("theta", theta)
  }
}
