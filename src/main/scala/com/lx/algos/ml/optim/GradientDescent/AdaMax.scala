package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, max, norm}
import breeze.numerics.{abs, sqrt}
import com.lx.algos.ml.MAX_DLOSS
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.norm.{DefaultNormFunction, L1NormFunction, L2NormFunction}
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{AutoGrad, SimpleAutoGrad}

import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 3:58 PM 25/12/2017
  */

/**
  * 该方法是adam的变种，只是在学习率上限上加了一个范围约束，详见<http://blog.csdn.net/u012759136/article/details/52302426>
  */
class AdaMax extends Adam {

  override def gamma: Double = {
    println(s"warning: gamma will be not  used in ${getClass.getSimpleName}")
    super.gamma
  }

  override def set_gamma(gamma: Double): AdaGrad = {
    println(s"warning: gamma will be not used in ${getClass.getSimpleName} though you set it!")
    super.set_gamma(gamma)
  }

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y_format = format_y(DenseMatrix(y).reshape(y.size, 1), loss)

    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_moment1 = DenseMatrix.zeros[Double](x.cols, 1) //一阶梯度累加
      var cache_moment2 = DenseMatrix.zeros[Double](x.cols, 1) //二阶梯度累加

      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        val batch_data: Seq[(DenseMatrix[Double], DenseMatrix[Double])] = get_minibatch(x, y_format, batchSize)

        for ((sub_x, sub_y) <- batch_data) {
          val autoGrad = new AutoGrad(sub_x, sub_y, _theta, loss, penaltyNorm, lambda)
          val grad = autoGrad.avgGrad //n*1 matrix

          cache_moment1 = beta1 * cache_moment1 + (1 - beta1) * grad
          cache_moment2 = max(beta2 * cache_moment2, abs(grad))

          val bias_moment1 = cache_moment1 / (1 - Math.pow(beta1, epoch))
          val bias_moment2 = cache_moment2

          _theta -= eta * bias_moment1 / (bias_moment2 + eps) / cache_moment2

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
