package com.lx.algos.optim

import breeze.linalg.{DenseVector, Matrix}
import breeze.numerics.sqrt
import com.lx.algos.MAX_DLOSS
import com.lx.algos.metrics.ClassificationMetrics

import scala.reflect.ClassTag
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

    val x = X.toDenseMatrix
    weight = DenseVector.rand[Double](x.cols) //init_weight

    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_grad = DenseVector.zeros[Double](x.cols)
      var cache_delateT = DenseVector.zeros[Double](x.cols)
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(weight) + intercept

          val y_format = if (y(i) == 1.0) 1.0 else -1.0 //需要注意，分类损失函数的格式化为-1和1

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          if (penalty == "l2") {
            l2penalty(Math.max(0, 1 - eta * lambda))
          } else{
            l1penalty(eta, lambda)
          }

          val grad = dloss * ele

          cache_grad =  gamma * cache_grad + (1 - gamma) * grad *:* grad
          val deltaT = - sqrt(cache_delateT + eps) / sqrt(cache_grad + eps) *:* grad
          cache_delateT = gamma * cache_delateT + (1 - gamma) * deltaT *:* deltaT

          weight += deltaT
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
