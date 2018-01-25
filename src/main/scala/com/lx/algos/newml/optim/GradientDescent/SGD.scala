package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.norm.{L2Norm, NormFunction}

/**
  *
  * @project scalaML
  * @author lx on 7:00 PM 11/01/2018
  */


class SGD(var lr: Double = 0.01,
          var momentum: Double = 0.9,
          var penalty: NormFunction = L2Norm,
          var nestrov: Boolean = true
         ) extends GDOptimizer {

  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    var velocity = variables.getParam[DenseMatrix[Double]]("velocity", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    if (nestrov) {
      velocity = momentum * velocity + grad
      theta -= lr * (grad + momentum * velocity)
    } else {
      velocity = momentum * velocity + grad * lr
      theta -= velocity
    }
    variables.setParam("velocity", velocity)
    variables.setParam("theta", theta)
  }
}
