package com.lx.algos.optim

import breeze.linalg.{DenseVector, Matrix}
import breeze.numerics.sqrt
import com.lx.algos.loss.LossFunction
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.utils.Param

import scala.reflect.ClassTag
import scala.util.control.Breaks.breakable

/**
  *
  * @project scalaML
  * @author lx on 5:05 PM 23/11/2017
  */


//非处理离散类型，如离散类型，需参照kaggle FTRL进行优化，未实现交叉版本的特征
class FTRL_Proximal(feature_size: Int) extends Optimizer with Param {

  override def setParam[T: ClassTag](name: String, value: T) = {
    super.setParam[T](name, value)
    this
  }

  override def setParams[T <: Iterable[(String, Any)]](params: T) = {
    super.setParams[T](params)
    this
  }

  protected def init_param(): FTRL_Proximal = {
    setParams(Seq(
      "verbose" -> false, //打印日志
      "printPeriod" -> 100,
      "iterNum" -> 1000,
      "alpha" -> 0.005, //learning rate
      "beta" -> 1.0, //smoothing parameter for adaptive learning rate
      "lambda1" -> 0.1, //L1 regularization, larger value means more regularized
      "lambda2" -> 5.0 //L2 regularization, larger value means more regularized
    ))
  }

  init_param()


  var D = feature_size + 1 // number of weights to use

  //squared sum of past gradients
  var N: DenseVector[Double] = DenseVector.zeros[Double](D)
  //adaptive lr
  var Z: DenseVector[Double] = DenseVector.rand[Double](D)
  //weights

  def update(x: DenseVector[Double], p: Double, y: Double): FTRL_Proximal = {

    val grad = loss.dLoss(p, y) * x

    val square_grad = grad *:* grad
    val sigma = (sqrt(N + square_grad) - sqrt(N)) / alpha
    N += square_grad
    Z += grad - sigma *:* _weight
    this
  }


  def loss = getParam[LossFunction]("loss")

  def set_loss(lossFunction: LossFunction) = setParam[LossFunction]("loss", lossFunction)

  def alpha = getParam[Double]("alpha")

  def set_alpha(alpha: Double) = setParam[Double]("alpha", alpha)

  def beta = getParam[Double]("beta")

  def set_beta(alpha: Double) = setParam[Double]("beta", beta)

  def lambda1 = getParam[Double]("lambda1")

  def set_lambda1(lambda1: Double) = setParam[Double]("lambda1", lambda1)

  def lambda2 = getParam[Double]("lambda2")

  def set_lambda2(lambda2: Double) = setParam[Double]("lambda2", lambda2)

  def iterNum = getParam[Int]("iterNum")

  def set_iterNum(iterNum: Int) = setParam[Int]("iterNum", iterNum)

  def verbose = getParam[Boolean]("verbose")

  def set_verbose(verbose: Boolean) = setParam[Boolean]("verbose", verbose)

  def printPeriod = getParam[Int]("printPeriod")

  def set_printPeriod(printPeriod: Int) = setParam[Int]("printPeriod", printPeriod)


  //仅用于训练过程中的预测
  private def _predict(x: DenseVector[Double]): Double = {
    // wTx is the inner product of w and x
    var wTx = 0.0

    var new_W = DenseVector.zeros[Double](D)
    for (i <- 0 until x.length) {
      val sign = if (Z(i) < 0) -1.0 else 1.0

      /** build w on the fly using z and n, hence the name - lazy weights
        * we are doing this at prediction instead of update time is because
        * this allows us for not storing the complete
        */
      new_W(i) = if (sign * Z(i) <= lambda1) 0.0
      else (sign * lambda1 - Z(i)) / ((beta + Math.sqrt(N(i))) / alpha + lambda2)

      wTx += new_W(i) * x(i)
    }

    _weight = new_W
    wTx
  }


  override def fit(X: Matrix[Double], y: Seq[Double]) = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix

    breakable {
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until X.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = _predict(ele)
          val y_format = format_y(y(i), loss)
          update(ele, y_pred, y_format)
          totalLoss += loss.loss(y_pred, y_format)
        }
        val avg_loss = totalLoss / x.rows

        if (verbose) {
          if (epoch % printPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, avg_loss)
          }
        }
      }
    }

    this

  }

  override def predict(X: Matrix[Double]) = super.predict_lr(X)

  override def predict_proba(X: Matrix[Double]) = super.predict_proba_lr(X)
}
