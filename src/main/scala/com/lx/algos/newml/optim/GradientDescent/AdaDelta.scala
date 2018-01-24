package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import breeze.numerics.sqrt
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.optim.Optimizer

/**
  *
  * @project scalaML
  * @author lx on 5:40 PM 24/01/2018
  */

class AdaDelta(var lr: Double = 0.01,
               var gamma: Double = 0.95, //公式中参数gamma
               var eps: Double = 1e-8
              ) extends Optimizer {
  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //梯度平方累加信息
    var grad2 = variables.getParam[DenseMatrix[Double]]("grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    //theta平方累加信息
    var deltaT2 = variables.getParam[DenseMatrix[Double]]("deltaT2", DenseMatrix.zeros[Double](grad.rows, grad.cols))

    grad2 = gamma * grad2 + (1 - gamma) * grad *:* grad
    val lr_grad = sqrt(deltaT2 + eps) / sqrt(grad2 + eps)
    val tmp_deltaT = lr_grad *:* grad
    deltaT2 = gamma * deltaT2 + (1 - gamma) * tmp_deltaT *:* tmp_deltaT

    theta -= tmp_deltaT
    variables.setParam("grad2", grad2)
    variables.setParam("deltaT2", deltaT2)
    variables.setParam("theta", theta)
  }
}
