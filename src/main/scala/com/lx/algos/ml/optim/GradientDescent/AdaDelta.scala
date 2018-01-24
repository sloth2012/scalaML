package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import breeze.numerics.sqrt
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{AutoGrad, SimpleAutoGrad}

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
    val y_format = format_y(DenseMatrix(y).reshape(y.size, 1), loss)

    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_grad = DenseMatrix.zeros[Double](x.cols, 1)
      var cache_deltaT = DenseMatrix.zeros[Double](x.cols, 1)

      for (epoch <- 1 to iterNum) {
        var totalLoss: Double = 0
        val batch_data: Seq[(DenseMatrix[Double], DenseMatrix[Double])] = get_minibatch(x, y_format, batchSize)

        for ((sub_x, sub_y) <- batch_data) {
          val autoGrad = new AutoGrad(sub_x, sub_y, _theta, loss, penaltyNorm, lambda)
          val grad = autoGrad.avgGrad //n*1 matrix

          cache_grad =  gamma * cache_grad + (1 - gamma) * grad *:* grad
          val lr_grad = sqrt(cache_deltaT + eps) / sqrt(cache_grad + eps)
          val deltaT = lr_grad *:* grad
          cache_deltaT = gamma * cache_deltaT + (1 - gamma) * deltaT *:* deltaT
          _theta -= deltaT

          autoGrad.updateTheta(_theta)
          totalLoss += autoGrad.totalLoss
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
