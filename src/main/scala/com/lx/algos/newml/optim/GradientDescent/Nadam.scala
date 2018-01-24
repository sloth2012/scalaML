package com.lx.algos.newml.optim.GradientDescent
import breeze.linalg.{DenseMatrix, max}
import breeze.numerics.{pow, sqrt}
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.optim.Optimizer

/**
  *
  * @project scalaML
  * @author lx on 6:51 PM 24/01/2018
  */

class Nadam(var lr: Double = 0.002,
            var beta: (Double, Double) = (0.99, 0.999),
            var eps: Double = 1e-8
           ) extends Optimizer {

  private def moment_schedule(time: Int): Double = (1 - 0.5 * pow(0.96, time / 250.0)) * beta._1

  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //梯度累加信息
    var grad1 = variables.getParam[DenseMatrix[Double]]("grad1", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    //梯度平方累加信息
    var grad2 = variables.getParam[DenseMatrix[Double]]("grad2", DenseMatrix.zeros[Double](grad.rows, grad.cols))
    //动量相乘信息
    var m_schedule = variables.getParam[Double]("m_schedule", 1)
    //timestamp信息
    val step = variables.getParam[Int]("step", 0)
    val scheduler_t = moment_schedule(epoch)
    val scheduler_t_1 = moment_schedule(epoch + 1)

    if(step < epoch) {
      m_schedule *= scheduler_t
      variables.setParam("step", epoch)
      variables.setParam("m_schedule", m_schedule)
    }

    val read_grad = grad / (1 - m_schedule)
    grad1 = beta._1 * grad1 + (1 - beta._1) * grad
    grad2 = beta._2 * grad2 + (1 - beta._2) * grad *:* grad

    val bias1 = grad1 / (1 - m_schedule * scheduler_t_1)
    val bias2 = grad2 / (1 - pow(beta._2, step))

    val avg_cache_moment1 = (1 - scheduler_t) * read_grad + scheduler_t_1 * bias1

    theta -= lr * avg_cache_moment1 / (sqrt(bias2) + eps)

    variables.setParam("grad1", grad1)
    variables.setParam("grad2", grad2)
    variables.setParam("theta", theta)
  }

}
