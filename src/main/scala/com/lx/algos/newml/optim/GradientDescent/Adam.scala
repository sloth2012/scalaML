package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, max}
import breeze.numerics.{pow, sqrt}
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.optim.Optimizer

/**
  *
  * @project scalaML
  * @author lx on 6:13 PM 24/01/2018
  */

class Adam(var lr: Double = 0.002,
           var beta: (Double, Double) = (0.9, 0.999),
           var amsgrad: Boolean = false,
           var eps: Double = 1e-8
          ) extends Optimizer {
  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //梯度累加信息
    var grad1 = variables.getParam[DenseMatrix[Double]]("grad1", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    //梯度平方累加信息
    var grad2 = variables.getParam[DenseMatrix[Double]]("grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))

    grad1 = beta._1 * grad1 + (1 - beta._1) * grad
    grad2 = beta._2 * grad2 + (1 - beta._2) * grad *:* grad

    val bias1 = grad1 / (1 - pow(beta._1, epoch))
    val bias2 = {
      if (amsgrad) {
        //最大梯度平方信息
        var max_grad2 = variables.getParam[DenseMatrix[Double]]("max_grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))
        max_grad2 = max(max_grad2, grad2)
        variables.setParam("max_grad2", max_grad2)
        max_grad2
      } else {
        grad2
      }
    } / (1 - pow(beta._2, epoch))

    theta -= lr * bias1 / (sqrt(bias2) + eps)

    variables.setParam("grad1", grad1)
    variables.setParam("grad2", grad2)
    variables.setParam("theta", theta)
  }
}
