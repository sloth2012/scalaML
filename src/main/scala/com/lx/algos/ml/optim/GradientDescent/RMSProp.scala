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
  * @author lx on 5:00 PM 21/11/2017
  */

class RMSProp extends AdaGrad {

  override protected def init_param(): RMSProp = {

    super.init_param()

    setParams(Seq(
      "gamma" -> 0.9, //梯度累加信息的衰减指数
      "eta" -> 0.001 //Hiton建议的设置
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
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(_weight)

          val y_format = format_y(y(i), loss)

          val autoGrad = new SimpleAutoGrad(ele, y_format, _weight, loss,  penaltyNorm, lambda)
          val grad = autoGrad.grad

          cache_grad =  gamma * cache_grad + (1 - gamma) * grad *:* grad
          val lr_grad = eta / sqrt(cache_grad + eps)

          _weight += -lr_grad *:* grad

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
