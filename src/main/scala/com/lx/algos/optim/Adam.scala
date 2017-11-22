package com.lx.algos.optim

import breeze.linalg.{DenseVector, Matrix}
import breeze.numerics.sqrt
import com.lx.algos.MAX_DLOSS
import com.lx.algos.metrics.ClassificationMetrics

import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 10:05 PM 21/11/2017
  */

class Adam extends AdaGrad {
  override protected def init_param(): Adam = {

    super.init_param()

    setParams(Seq(
      "eta" -> 0.002, //梯度累加信息的衰减指数
      "beta1" -> 0.9,
      "beta2" -> 0.99
    ))

    this
  }

  init_param()

  def beta1 = getParam[Double]("beta1")

  def set_beta1(beta1: Double) = setParam[Double]("beta1", beta1)

  def beta2 = getParam[Double]("beta2")

  def set_beta2(beta1: Double) = setParam[Double]("beta2", beta2)

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    val x = X.toDenseMatrix
    weight = DenseVector.rand[Double](x.cols) //init_weight

    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_moment1 = DenseVector.zeros[Double](x.cols)
      var cache_moment2 = DenseVector.zeros[Double](x.cols)

      var t = 0
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          t += 1
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(weight) + intercept

          val y_format = if (y(i) == 1.0) 1.0 else -1.0 //需要注意，分类损失函数的格式化为-1和1

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          if (penalty == "l2") {
            l2penalty(Math.max(0, 1 - eta * lambda))
          } else {
            l1penalty(eta, lambda)
          }

          val grad = dloss * ele

          cache_moment1 = beta1 * cache_moment1 + (1 - beta1) * grad
          cache_moment2 = beta2 * cache_moment2 + (1 - beta2) * grad *:* grad

          val bias_moment1 = cache_moment1 / (1 - Math.pow(beta1, t))
          val bias_moment2 = cache_moment2 / (1 - Math.pow(beta2, t))

          weight += -eta * bias_moment1 / sqrt(bias_moment2 + eps)
          //          intercept += -eta * dloss

          totalLoss += loss.loss(y_pred, y_format)
        }
        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if (epoch % printPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, avg_loss)
          }
        }
        if (converged) {
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
