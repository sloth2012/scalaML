package com.lx.algos.newml.optim.newton

import breeze.linalg.DenseMatrix
import breeze.numerics.abs
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.optim.newton.Interpolation.Method
import com.lx.algos.newml.utils.TimeUtil

import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 4:29 PM 25/01/2018
  */

sealed trait LineSearch {
  def getStep(autoGrad: AutoGrad, dk: DenseMatrix[Double]): Double
}

//一种不精确一维搜索实现方法，满足WolfePowell准则
//参照<http://blog.csdn.net/mytestmy/article/details/16903537>
class WolfePowell(var c1: Double = 0.1, //通常在(0,0.5)之间,指的是ρ
                  var c2: Double = 0.4, //0.1相当于线性搜索，0.9相当于弱的线性搜索，通常取0.4，其应该在(rho,1)之间，指σ
                  var init_interval: (Double, Double) = (0, 1000), //初始化搜索区间
                  var eps: Double = 0.01, //区间收敛值
                  var interpolationMethod: Method = Interpolation.TWO_POINT_QUADRATIC
                 ) extends LineSearch {

  def zoom(alpha_lo: Double,
           alpha_hi: Double,
           grad_zero: Double,
           f_zero: Double,
           autoGrad: AutoGrad,
           theta: DenseMatrix[Double],
           dk: DenseMatrix[Double]): Double = {
    var loop = 1
    var low = alpha_lo
    var high = alpha_hi
    var alpha = low

    var last_alpha = low //用于排除多次更新alpha收敛
    var converged_counter = 0
    val counterMax = 10

    val flat_dk = dk.reshape(dk.size, 1)

    breakable {
      while (loop > 0) {
//        println(s"zoom loop $loop is alpha($alpha) in [$low, $high]")
        //防止区间过小
        if (abs(high - low) < eps || converged_counter > counterMax) {
          break
        }
        if(abs(last_alpha - alpha) < eps){
          converged_counter += 1
        }else{
          converged_counter = 0
        }

        //初始化，需要用到插值方法，这里用了二点二次插值
        val low_grad = autoGrad.updateTheta(theta + low * dk).grad
        val low_loss = autoGrad.loss
        val high_loss = autoGrad.updateTheta(theta + high * dk).loss
        interpolationMethod match {
          case Interpolation.TWO_POINT_QUADRATIC => {
            val value: Double = (low_grad.reshape(low_grad.size, 1).t * flat_dk).data(0)
            alpha = low - 0.5 * (low - high) / (1 - (low_loss - high_loss) /
              (value * (low - high)))
          }
          case Interpolation.BISECTION => alpha = (low + high) / 2
        }
        val f = autoGrad.updateTheta(theta + alpha * dk).loss
        //可能会一直陷入到该分支循环，通过last_alpha控制
        if (f > f_zero + c1 * alpha * grad_zero || f >= low_loss) {
          high = alpha
        }
        else {
          val grad_new: Double = (autoGrad.grad.reshape(dk.size, 1).t * flat_dk).data(0)
          if (abs(grad_new) <= c2 * abs(grad_zero)) {
            break
          }
          if (grad_new * (high - low) >= 0) {
            high = low
          }
          low = alpha
        }

        loop += 1
      }
    }
    alpha
  }

  override def getStep(autoGrad: AutoGrad, dk: DenseMatrix[Double]): Double = {
    val rnd = new Random(TimeUtil.currentMillis)
    val alpha1 = init_interval._1
    val alpha2 = init_interval._2 //最大搜索步长，这里设置的较大

    var alpha = rnd.nextDouble() * alpha2

    var loop = 1

    val J = autoGrad.loss //记录第一次最初的损失值
    var f = J
    var alpha_last = alpha1
    val grad = autoGrad.grad

    val flat_dk = dk.reshape(dk.size, 1)
    val grad_zero = (grad.reshape(grad.size, 1).t * flat_dk).data(0) // 记录最初的g(0)

    val theta = autoGrad.theta

    breakable {
      while (loop > 0) {
        //        println(s"getStep loop $loop is alpha($alpha) in [$alpha1, $alpha2]")
        val f_new = autoGrad.updateTheta(theta + alpha * dk).loss

        //J means f0
        if (f_new > J + (c1 * alpha * grad_zero) || (loop > 1 && f_new >= f)) {
          alpha = zoom(alpha_last, alpha, grad_zero, J, autoGrad, theta, dk)
          break
        }

        val grad_new: Double = (autoGrad.grad.reshape(grad.size, 1).t * flat_dk).data(0)
        if (grad_new <= c2 * abs(grad_zero)) {
          break
        }
        if (grad_new >= 0) {
          alpha = zoom(alpha, alpha_last, grad_zero, J, autoGrad, theta, dk)
          break
        }

        f = f_new
        alpha_last = alpha

        alpha = alpha + (alpha2 - alpha) * rnd.nextDouble()

        loop += 1
      }
    }

    alpha
  }
}


object LineSearch {
  val wolfePowell = new WolfePowell

}