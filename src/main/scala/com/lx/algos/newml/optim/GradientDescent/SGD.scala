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
          var nestrov: Boolean = false,
          var earlyStop: Boolean = true,
          var eps: Double = 1e-5 //迭代loss收敛约束,配合earlyStop
         ) extends Optimizer {



  override def run(autograd: AutoGrad, epoch: Int): Unit = {
    val grad = autograd.grad
    val totalLoss = variables.getParam[Double]("totalLoss", 0)
    val runsamples = variables.getParam[Double]("samples", 0)
    val theta = variables.getParam[DenseMatrix[Double]]("theta", autograd.theta)


    val last_epoch = variables.getParam[Int]("epoch", 1)

    if(last_epoch < epoch){

      val avg_loss = totalLoss / runsamples
      val last_avg_loss = variables.getParam[Double]("avg_loss", 0)

      if(earlyStop && abs(avg_loss - last_avg_loss) < eps) {
        println(s"the optimizer converged in epoch $epoch!")
      }else {
        variables.setParam("epoch", epoch)
      }
    }else {

      val velocity = variables.getParam[DenseMatrix[Double]]("velocity", DenseMatrix.zeros[Double](grad.rows, grad.cols))
      val (new_velocity, new_theta) = if (nestrov) {
        val new_velocity: DenseMatrix[Double] = momentum * velocity + grad
        val new_theta = theta - lr * (grad + momentum * new_velocity) //未改变对象，是对象的引用，值的改变

        (new_velocity, new_theta)
      } else {
        val new_velocity: DenseMatrix[Double] = momentum * velocity + grad * lr
        val new_theta = theta - new_velocity

        (new_velocity, new_theta)
      }

      var new_totalLoss = totalLoss + autograd.updateTheta(new_theta).totalLoss
      var new_runsamples = runsamples + autograd.size
      variables.setParam("velocity", new_velocity)
      variables.setParam("theta", new_theta)
      variables.setParam("totalLoss", new_totalLoss)
      variables.setParam("samples", new_runsamples)
    }
  }
}
