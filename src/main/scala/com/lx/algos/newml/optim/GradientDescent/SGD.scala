package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import breeze.numerics.abs
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
          var nestrov: Boolean = true,
          var earlyStop: Boolean = true,
          var eps: Double = 1e-5 //迭代loss收敛约束,配合earlyStop
         ) extends Optimizer {


  override def run(autograd: AutoGrad, epoch: Int): Unit = {
    val grad = autograd.grad
    var totalLoss = variables.getParam[Double]("totalLoss", 0)
    var runsamples = variables.getParam[Double]("samples", 0)
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autograd.theta)


    val last_epoch = variables.getParam[Int]("epoch", 1)

    if (last_epoch < epoch) {
      val avg_loss = totalLoss / runsamples
      val last_avg_loss = variables.getParam[Double]("avg_loss", 0)

      if (earlyStop && abs(avg_loss - last_avg_loss) < eps) {
        println(s"the optimizer converged in epoch ${epoch - 1}!") //上一轮结束后收敛的
        variables.setParam("converged", true)
      } else {
        variables.setParam("epoch", epoch)
        variables.setParam("avg_loss", avg_loss)
      }
      runsamples = 0
      totalLoss = 0
    }

    val converged = variables.getParam[Boolean]("converged", false)
    if (converged == false) {

      var velocity = variables.getParam[DenseMatrix[Double]]("velocity", DenseMatrix.zeros[Double](grad.rows, grad.cols))
      if (nestrov) {
        velocity = momentum * velocity + grad
        theta -= lr * (grad + momentum * velocity)
      } else {
        velocity = momentum * velocity + grad * lr
        theta  -= velocity
      }

      totalLoss += autograd.updateTheta(theta).loss * autograd.size
      runsamples += autograd.size
      variables.setParam("velocity", velocity)
      variables.setParam("theta", theta)
      variables.setParam("totalLoss", totalLoss)
      variables.setParam("samples", runsamples)
    }
  }
}
