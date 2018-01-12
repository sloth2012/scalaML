package com.lx.algos.newml.optim.GradientDescent

import breeze.linalg.DenseMatrix
import com.lx.algos.ml.utils.AutoGrad
import com.lx.algos.newml.norm.{L2Norm, NormFunction}
import com.lx.algos.newml.optim.Optimizer
import com.lx.algos.newml.utils.Param

/**
  *
  * @project scalaML
  * @author lx on 7:00 PM 11/01/2018
  */


class SGD(var lr: Double = 0.01,
          var momentum: Double = 0.9,
          var penalty: NormFunction = L2Norm,
          var nestrov: Boolean = false,
          var verbose: Boolean = false, //是否打印日志
          var logPeriod: Int = 100 //打印周期，只在verbose为true时有效
         ) extends Optimizer {


  def init() = {

  }

  override def run(autograd: AutoGrad): Double = {
    val grad = autograd.avgGrad

    val velocity = variables.getParam[DenseMatrix[Double]]("velocity")
    val theta = variables.getParam[DenseMatrix[Double]]("theta")
    if (nestrov) {
      val new_velocity = momentum * velocity - lr * grad
      val delta = momentum * momentum * velocity - (1 + momentum) * lr * grad

      theta += delta //未改变对象，是对象的引用，值的改变
      variables.setParam("velocity", new_velocity)
    } else {
      val new_velocity = momentum * velocity + grad * lr
      theta -= new_velocity

      variables.setParam("velocity", new_velocity)
    }

    0

  }
}
