package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseVector, Matrix}
import breeze.numerics.sqrt
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.SimpleAutoGrad

import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 9:09 PM 21/11/2017
  */

class AdaDelta extends AdaGrad{

  override protected def init_param(): AdaDelta = {

    super.init_param()

    setParams(Seq(
      "gamma" -> 0.95 //梯度累加信息的衰减指数
    ))

    this
  }

  init_param()

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix

    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_grad = DenseVector.zeros[Double](x.cols)
      var cache_delateT = DenseVector.zeros[Double](x.cols)
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(_weight)

          val y_format = format_y(y(i), loss)

          val autoGrad = new SimpleAutoGrad(ele, y_format, _weight, loss,  penaltyNorm, lambda)
          val grad = autoGrad.grad

          cache_grad =  gamma * cache_grad + (1 - gamma) * grad *:* grad
          val lr_grad = sqrt(cache_delateT + eps) / sqrt(cache_grad + eps)
          val deltaT = -lr_grad *:* grad
          cache_delateT = gamma * cache_delateT + (1 - gamma) * deltaT *:* deltaT

          _weight += deltaT

          autoGrad.updateTheta(_weight)
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
