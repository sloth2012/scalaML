package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.norm.{L2Norm, NormFunction}
import com.lx.algos.newml.optim.Optimizer

/**
  *
  * @project scalaML
  * @author lx on 7:00 PM 11/01/2018
  */


class SGD(var lr: Double = 0.01,
          var momentum: Double = 0.9,
          var penalty: NormFunction = L2Norm,
          var nestrov: Boolean = true
         ) extends Optimizer {

  override def run(autograd: AutoGrad, epoch: Int): Unit = {
    val grad = autograd.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autograd.theta)

    var velocity = variables.getParam[DenseMatrix[Double]]("velocity", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    if (nestrov) {
      velocity = momentum * velocity + grad
      theta -= lr * (grad + momentum * velocity)
    } else {
      velocity = momentum * velocity + grad * lr
      theta -= velocity //theta直接使用了对象的引用，改变了内部的值，因此不用回写到velocity
    }
    variables.setParam("velocity", velocity)
    variables.setParam("theta", theta)
  }
}
