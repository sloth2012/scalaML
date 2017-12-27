package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseVector, Matrix}
import breeze.numerics.{pow, sqrt}
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.SimpleAutoGrad

import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 3:09 PM 26/12/2017
  */

class Nadam extends Adam {

  override protected def init_param() = {
    super.init_param()

    setParams(Seq(
      "beta1" -> 0.99
    ))

    this
  }

  init_param

  def moment_schedule(time: Int = t): Double = (1 - 0.5 * pow(0.96, time / 250.0)) * beta1

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix

    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_moment1 = DenseVector.zeros[Double](x.cols) //一阶梯度累加
      var cache_moment2 = DenseVector.zeros[Double](x.cols) //二阶梯度累加
      var moment_schedule_sum = 0.0

      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          t += 1

          val cache_scheduler_t = moment_schedule(t)
          val cache_scheduler_t_1 = moment_schedule(t + 1)
          moment_schedule_sum *= cache_scheduler_t
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(_theta.toDenseVector)

          val y_format = format_y(y(i), loss)

          val autoGrad = new SimpleAutoGrad(ele, y_format, _theta, loss, penaltyNorm, lambda)

          val grad = autoGrad.grad

          val real_grad = grad / (1 - moment_schedule_sum)

          cache_moment1 = beta1 * cache_moment1 + (1 - beta1) * grad
          cache_moment2 = beta2 * cache_moment2 + (1 - beta2) * grad *:* grad

          val bias_moment1 = cache_moment1 / (1 - moment_schedule_sum * cache_scheduler_t_1)
          val bias_moment2 = cache_moment2 / (1 - Math.pow(beta2, t))

          val avg_cache_moment1 = (1 - cache_scheduler_t) * real_grad + cache_scheduler_t_1 * bias_moment1


          _theta += (-eta * avg_cache_moment1 / (sqrt(bias_moment2) + eps)).toDenseMatrix.reshape(_theta.rows, 1)

          autoGrad.updateTheta(_theta)
          totalLoss += autoGrad.loss
        }
        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if (epoch % printPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, avg_loss)
          }
        }
        if (converged && early_stop) {
          println(s"converged at iter $epoch!")
          val acc = ClassificationMetrics.accuracy_score(predict(X), y)
          log_print(epoch, acc, avg_loss)
          break
        }

        last_avg_loss = avg_loss
      }
    }

    this
  }
}
