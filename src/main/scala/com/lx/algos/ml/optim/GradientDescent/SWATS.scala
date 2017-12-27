package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, max}
import breeze.numerics.{abs, sqrt}
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{AutoGrad, SimpleAutoGrad}

import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 11:32 AM 27/12/2017
  */

//最新2017 ICLR 2018高分论文的adam和sgd的融合切换方法
//<https://mp.weixin.qq.com/s/alRLnkSAW2r-bvyFxert7Q>
//<https://arxiv.org/abs/1712.07628>
class SWATS extends Adam {

  override protected def init_param() = {
    super.init_param()

    setParams(Seq(
      "eta" -> 1e-3,
      "lr_decay_epoch" -> 150, //学习率decay周期
      "lr_decay_ratio" -> 0.1, //衰减系数
      "early_stop" -> true //是否收敛自动停止
    ))

    this
  }

  init_param

  def lr_decay_epoch: Int = getParam[Int]("lr_decay_epoch")

  def set_lr_decay_epoch(epoch: Int) = setParam[Int]("lr_decay_epoch", epoch)

  def lr_decay_ratio: Double = getParam[Double]("lr_decay_ratio")

  def set_lr_decay_ratio(ratio: Double) = setParam[Double]("lr_decay_ratio", ratio)

  override lazy val eps = 1e-9

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y_format = format_y(DenseMatrix(y).reshape(y.size, 1), loss)

    breakable {

      var useAdam = true

      var last_avg_loss = Double.MaxValue
      var m_k = DenseMatrix.zeros[Double](x.cols, 1) //一阶梯度累加
      var a_k = DenseMatrix.zeros[Double](x.cols, 1) //二阶梯度累加

      var r_k = 0.0
      var lr_k = eta
      var lambda_k = 0.0
      var v_k = DenseMatrix.zeros[Double](x.cols, 1)
      var adam_lr = 0.0
      for (epoch <- 1 to iterNum) {
        if (epoch % lr_decay_epoch == 0) lr_k *= lr_decay_ratio

        var totalLoss: Double = 0

        val batch_data: Seq[(DenseMatrix[Double], DenseMatrix[Double])] = get_minibatch(x, y_format, batchSize)

        for ((sub_x, sub_y) <- batch_data) {
          val autoGrad = new AutoGrad(sub_x, sub_y, _theta, loss, penaltyNorm, lambda)
          val grad = autoGrad.avgGrad //n*1 matrix

          if (useAdam == false) {
            v_k = beta1 * v_k + grad
            _theta -= ((1 - beta1) * adam_lr * v_k).toDenseMatrix.reshape(_theta.rows, 1)
          } else {


            m_k = beta1 * m_k + (1 - beta1) * grad
            a_k = beta2 * a_k + (1 - beta2) * grad *:* grad

            val p_k = -lr_k * sqrt(1 - -Math.pow(beta2, epoch)) / (1 - Math.pow(beta1, epoch)) * m_k / (sqrt(a_k) + eps)

            _theta += p_k

            val z = (p_k.t * grad).data(0)

            if (z != 0) {
              r_k = -(p_k.t * p_k).data(0) / z

              lambda_k = beta2 * lambda_k + (1 - beta2) * r_k

              val delta_lr = lambda_k / (1 - Math.pow(beta2, epoch))

              //尝试了下，发现切换学习率的时候，应该加一个约束，即delta_lr>0，否则采用sgd的时候学习率变成负的了，每次都变成累加梯度了。
              if (epoch > 1 && delta_lr > 0 && abs(delta_lr - r_k) < eps) {
                useAdam = false
                adam_lr = delta_lr
                println(s"now use sgd from epoch ${epoch + 1}, lr is $adam_lr!")
              }
            }
          }
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
