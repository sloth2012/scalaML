package com.lx.algos.newml.optim.newton

import breeze.linalg.DenseMatrix
import breeze.numerics.abs
import com.lx.algos.newml.autograd.AutoGrad

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
                  var alpha0: Double = 1, //初始化alpha
                  var alpha_min: Double = 1e-5, //alpha最小值约束
                  var eps: Double = 1e-8, //区间收敛值
                  var maxIterNum: Int = 100
                 ) extends LineSearch {


  def isInInterval(value: Double, range: (Double, Double)) = {
    val (a, b) = if (range._1 > range._2) range.swap else range

    value >= a && value <= b
  }


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


    //        if(alpha_lo > alpha_hi) {
    //          throw new RuntimeException(s"Invalid interval [$low, $high] of stepsize in zoom procedure")
    //        }

    var last_alpha = low //用于排除多次更新alpha收敛
    var converged_counter = 0
    val counterMax = 10

    val flat_dk = dk.reshape(dk.size, 1)


    breakable {
      while (loop < maxIterNum && abs(high - low) >= eps) {
        println(s"zoom loop $loop is alpha($alpha) in [$low, $high]")

        val low_loss = autoGrad.updateTheta(theta + low * dk).loss

        if (abs(last_alpha - alpha) < eps) {
          converged_counter += 1
        } else {
          converged_counter = 0
        }
        last_alpha = alpha
        if (converged_counter > counterMax) {
          alpha = Interpolation.bisection.getValue(low, high)
        } else {
          val low_grad = (autoGrad.grad.reshape(dk.size, 1).t * flat_dk).data(0)
          val high_loss = autoGrad.updateTheta(theta + high * dk).loss

          alpha = Interpolation.quadratic.getValue(low, low_loss, high, high_loss, low_grad)
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

    val grad = autoGrad.grad
    val theta = autoGrad.theta

    val flat_dk = dk.reshape(dk.size, 1)
    val grad_zero = (grad.reshape(grad.size, 1).t * flat_dk).data(0) // 记录最初的g(0)
    val f_zero = autoGrad.loss //记录第一次最初的损失值,f_zero

    var f_last = f_zero
    var alpha_last: Double = 0

    //采用二次插值来寻找切换点
    val interpolation = new QuadraticInterpolation

    var alpha = alpha0

    var loop = 1
    breakable {
      while (loop < maxIterNum) {
        val f = autoGrad.updateTheta(theta + alpha * dk).loss
        val grad_new: Double = (autoGrad.grad.reshape(grad.size, 1).t * flat_dk).data(0)

        //f_zero means f0
        if (f > f_zero + (c1 * alpha * grad_zero) || (loop > 1 && f >= f_last)) {
          alpha = zoom(alpha_last, alpha, grad_zero, f_zero, autoGrad, theta, dk)
          break
        }

        if (abs(grad_new) <= c2 * abs(grad_zero)) {
          break
        }
        if (grad_new >= 0) {
          alpha = zoom(alpha, alpha_last, grad_zero, f_zero, autoGrad, theta, dk)
          break
        }

        alpha = interpolation.getValue(0, f_zero, alpha, f, grad_zero) //x1, f1, x2, f2, f1'

        //        if(isInInterval(alpha, (alpha_min, al)))
        println(s"getStep loop $loop is alpha($alpha)")

        f_last = f
        alpha_last = alpha

        loop += 1
      }
    }

    if (loop >= maxIterNum) alpha = alpha_min //未找到合适的，就返回一个最小的

    alpha
  }
}


class GoldSection(var maxIterNum: Int = 100,
                  var eps: Double = 1e-5
                 ) extends LineSearch {
  override def getStep(autoGrad: AutoGrad, dk: DenseMatrix[Double]): Double = {
    val theta = autoGrad.theta

    var h = Math.random() //步长
    var alpha = 0.0 //init alpha

    var (alpha1, alpha2) = (alpha, h)
    var (theta1, theta2) = (theta + alpha1 * dk, theta + alpha2 * dk)

    var f1 = autoGrad.updateTheta(theta1).loss
    var f2 = autoGrad.updateTheta(theta2).loss

    //######################################
    // 进退法
    //######################################
    var loop = 1
    var a, b = 0.0
    breakable {

      //进退法找区间
      while (loop < maxIterNum) {

        if (f1 > f2) h *= 2
        else {
          h *= -1

          val tmp_alpha = alpha1
          alpha1 = alpha2
          alpha2 = tmp_alpha

          val tmp_f = f1
          f1 = f2
          f2 = tmp_f
        }

        val alpha3 = alpha2 + h
        val theta3 = theta + alpha3 * dk
        val f3 = autoGrad.updateTheta(theta3).loss

        //            println(s"f3 - f2 is ${f3 - f2}")

        if (f3 > f2) {
          a = Math.min(alpha1, alpha3)
          b = Math.max(alpha1, alpha3)
          break
        } else {
          alpha1 = alpha2
          alpha2 = alpha3

          f1 = f2
          f2 = f3
        }

        loop += 1
      }
    }

//    println(s"find best interval with loop $loop is [$a, $b]")
    //######################################
    // 黄金分割法
    //######################################
    loop = 0
    breakable {
      // 黄金分割法找步长
      while (loop < maxIterNum) {
        alpha1 = a + 0.382 * (b - a)
        alpha2 = a + 0.618 * (b - a)

        theta1 = theta + alpha1 * dk
        theta2 = theta + alpha2 * dk

        f1 = autoGrad.updateTheta(theta1).loss
        f2 = autoGrad.updateTheta(theta2).loss

        if (f1 > f2) a = alpha1
        if (f1 < f2) b = alpha2

        if (abs(a - b) <= eps) {
          alpha = Interpolation.bisection.getValue(a, b) //二分插值
          break
        }

        loop += 1
      }
    }
//    println(s"find best alpha with loop $loop is $alpha")
    alpha
  }
}

object LineSearch {
  val wolfePowell = new WolfePowell
  val goldSection = new GoldSection

}