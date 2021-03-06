package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import breeze.numerics.sqrt
import com.lx.algos.ml.loss.{LogLoss, LossFunction}
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{AutoGrad, Param, SimpleAutoGrad}
import com.lx.algos.ml.norm.{DefaultNormFunction, L1NormFunction, L2NormFunction}

import scala.reflect.ClassTag
import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 3:57 PM 21/11/2017
  */


class AdaGrad extends Optimizer with Param {


  protected def init_param(): AdaGrad = {
    setParams(Seq(
      "eta" -> 0.01, //learning_rate
      "lambda" -> 0.15, // 正则化权重,weigjht decay
      "verbose" -> false, //打印日志
      "printPeriod" -> 100,
      "penalty" -> "l2", //正则化系数，暂只实现l2
      "iterNum" -> 1000, //迭代轮数
      "loss" -> new LogLoss,
      "batchSize" -> 128 //minibatch size，一个batch多少样本
    ))

    this
  }


  override def setParam[T: ClassTag](name: String, value: T) = {
    super.setParam[T](name, value)
    this
  }

  override def setParams[T <: Iterable[(String, Any)]](params: T) = {
    super.setParams[T](params)
    this
  }

  init_param()

  def penaltyNorm = penalty.toLowerCase match {
    case "l1" => L1NormFunction
    case "l2" => L2NormFunction
    case _ => DefaultNormFunction
  }

  lazy val eps = 1e-8

  def batchSize = getParam[Int]("batchSize")

  def loss = getParam[LossFunction]("loss")

  def penalty = getParam[String]("penalty")

  def iterNum = getParam[Int]("iterNum")

  def printPeriod = getParam[Int]("printPeriod")

  def verbose = getParam[Boolean]("verbose")

  def eta = getParam[Double]("eta")

  def lambda = getParam[Double]("lambda")

  def gamma = getParam[Double]("gamma")

  def early_stop: Boolean = getParam[Boolean]("early_stop", true)

  def set_early_stop(early_stop: Boolean) = setParam[Boolean]("early_stop", early_stop)

  def set_gamma(gamma: Double) = setParam[Double]("gamma", gamma)

  def set_batchSize(batchSize: Int) = setParam[Int]("batchSize", batchSize)

  def set_loss(lossFunction: LossFunction) = setParam[LossFunction]("loss", lossFunction)

  def set_penalty(penalty: String) = setParam[String]("penalty", penalty)

  def set_iterNum(iterNum: Int) = setParam[Int]("iterNum", iterNum)

  def set_printPeriod(printPeriod: Int) = setParam[Int]("printPeriod", printPeriod)

  def set_verbose(verbose: Boolean) = setParam[Boolean]("verbose", verbose)

  def set_eta(eta: Double) = setParam[Double]("eta", eta)

  def set_lambda(lambda: Double) = setParam[Double]("lambda", lambda)


  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y_format = format_y(DenseMatrix(y).reshape(y.size, 1), loss)


    breakable {
      var last_avg_loss = Double.MaxValue
      var cache_grad = DenseMatrix.zeros[Double](x.cols, 1)

      for (epoch <- 1 to iterNum) {
        var totalLoss: Double = 0

        val batch_data: Seq[(DenseMatrix[Double], DenseMatrix[Double])] = get_minibatch(x, y_format, batchSize)

        for ((sub_x, sub_y) <- batch_data) {
          val autoGrad = new AutoGrad(sub_x, sub_y, _theta, loss, penaltyNorm, lambda)
          val grad = autoGrad.avgGrad //n*1 matrix

          cache_grad += grad *:* grad
          val lr_grad = eta / sqrt(cache_grad + eps)
          _theta -= lr_grad *:* grad

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

  override def predict(X: Matrix[Double]): Seq[Double] = {
    assert(weight != null)
    predict_lr(X)
  }

  override def predict_proba(X: Matrix[Double]): Seq[Double] = predict_proba_lr(X)
}
