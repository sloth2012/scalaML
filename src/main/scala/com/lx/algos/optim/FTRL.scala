package com.lx.algos.optim

import breeze.linalg.{DenseVector, Matrix, SparseVector}
import breeze.numerics.sqrt
import breeze.util.Sorting
import com.lx.algos.MAX_DLOSS
import com.lx.algos.loss.{LogLoss, LossFunction}
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.utils.Param

import scala.util.control.Breaks.{break, breakable}
import scala.collection.mutable.Map
import scala.reflect.ClassTag

/**
  *
  * @project scalaML
  * @author lx on 5:05 PM 23/11/2017
  */


//非处理离散类型，如离散类型，需参照kaggle FTRL进行优化
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
      "iterNum" -> 5000
    ))
  }

  init_param()


  def loss = getParam[LossFunction]("loss")

  def set_loss(lossFunction: LossFunction) = setParam[LossFunction]("loss", lossFunction)


  val alpha = 0.001 //learning rate
  val beta = 1.0 //smoothing parameter for adaptive learning rate
  val L1 = 0 //L1 regularization, larger value means more regularized
  val L2 = 0 //L2 regularization, larger value means more regularized

  var D = feature_size + 1// number of weights to use

  var interaction = false
  //squared sum of past gradients
  var N: DenseVector[Double] = DenseVector.zeros[Double](D)
  //adaptive lr
  var Z: DenseVector[Double] = DenseVector.rand[Double](D)
  //weights
  var W = _weight

  def update(x: DenseVector[Double], p: Double, y: Double): FTRL_Proximal = {

    val grad = loss.loss(p, y) * x

    val square_grad = grad *:* grad
    val sigma = (sqrt(N + square_grad) - sqrt(N)) / alpha
    N += square_grad
    Z += grad - sigma *:* W
    this
  }

  def predict(x: DenseVector[Double]): Double = {
    // wTx is the inner product of w and x
    var wTx = 0.0

    for (i <- 0 until x.length) {
      val sign = if(Z(i) < 0) -1.0 else  1.0

      /** build w on the fly using z and n, hence the name - lazy weights
        * we are doing this at prediction instead of update time is because
        * this allows us for not storing the complete
        */
      W(i) = if (sign * Z(i) <= L1) 0.0
      else (sign * L1 - Z(i)) / ((beta + Math.sqrt(N(i))) / alpha + L2)

      wTx += W(i) * x(i)
    }

//    1./(1 + Math.exp(-Math.max(Math.min(wTx, 35), -35)))
    wTx
  }


//  def alpha = getParam[Double]("alpha")
//
//  def beta = getParam[Double]("beta")
//
//  def lambda1 = getParam[Double]("lambda1")
//
//  def lambda2 = getParam[Double]("lambda2")

  def iterNum = getParam[Int]("iterNum")
  def set_iterNum(iterNum: Int) = setParam[Int]("iterNum", iterNum)

  def verbose = getParam[Boolean]("verbose")
  def set_verbose(verbose: Boolean) = setParam[Boolean]("verbose", verbose)

  def printPeriod = getParam[Int]("printPeriod")
  def set_printPeriod(printPeriod: Int) = setParam[Int]("printPeriod", printPeriod)

//  def set_alpha(alpha: Double) = setParam[Double]("alpha", alpha)
//
//  def set_beta(alpha: Double) = setParam[Double]("beta", beta)
//
//  def set_lambda1(lambda1: Double) = setParam[Double]("lambda1", lambda1)
//
//  def set_lambda2(lambda2: Double) = setParam[Double]("lambda2", lambda2)


  override def fit(X: Matrix[Double], y: Seq[Double]) = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    W = _weight

    breakable {
      for (epoch <- 1 to 100) {

        var totalLoss: Double = 0

        for (i <- 0 until X.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = predict(ele)
          val y_format = if (y(i) == 1.0) 1.0 else -1.0 //需要注意，分类损失函数的格式化为-1和1

//          println(y_pred, y_format)

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
