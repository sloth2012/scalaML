package com.lx.algos.optim

import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import com.lx.algos.MAX_DLOSS
import com.lx.algos.loss.{LogLoss, LossFunction}
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.utils.{MatrixTools, Param}

import scala.reflect.ClassTag
import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 1:59 PM 20/11/2017
  */

//随机梯度下降
class SGD extends Optimizer with Param {


  protected def init_param(): SGD = {
    setParams(Seq(
      "eta" -> 0.01, //learning_rate
      "lambda" -> 0.15, // 正则化权重,weigjht decay
      "gamma" -> 0.9, //动量参数
      "nesterov" -> false, // NAG支持，改良版，需要配合gamma参数
      "verbose" -> false, //打印日志
      "printPeriod" -> 100,
      "penalty" -> "l2", //正则化系数，暂只实现l2
      "iterNum" -> 1000, //迭代轮数
      "loss" -> new LogLoss,
      "batchSize" -> 128 //minibatch size，一个batc多少样本
    ))

    this
  }

  init_param()

  override def setParam[T: ClassTag](name: String, value: T) = {
    super.setParam[T](name, value)
    this
  }

  override def setParams[T <: Iterable[(String, Any)]](params: T) = {
    super.setParams[T](params)
    this
  }

  def batchSize = getParam[Int]("batchSize")

  def loss = getParam[LossFunction]("loss")

  def penalty = getParam[String]("penalty")

  def iterNum = getParam[Int]("iterNum")

  def printPeriod = getParam[Int]("printPeriod")

  def verbose = getParam[Boolean]("verbose")

  def eta = getParam[Double]("eta")

  def lambda = getParam[Double]("lambda")

  def gamma = getParam[Double]("gamma")

  def nesterov = getParam[Boolean]("nesterov")


  def set_batchSize(batchSize: Int) = setParam[Int]("batchSize", batchSize)

  def set_loss(lossFunction: LossFunction) = setParam[LossFunction]("loss", lossFunction)

  def set_penalty(penalty: String) = setParam[String]("penalty", penalty)

  def set_iterNum(iterNum: Int) = setParam[Int]("iterNum", iterNum)

  def set_printPeriod(printPeriod: Int) = setParam[Int]("printPeriod", printPeriod)

  def set_verbose(verbose: Boolean) = setParam[Boolean]("verbose", verbose)

  def set_eta(eta: Double) = setParam[Double]("eta", eta)

  def set_lambda(lambda: Double) = setParam[Double]("lambda", lambda)

  def set_gamma(gamma: Double) = setParam[Double]("gamma", gamma)

  def set_nesterov(nesterov: Boolean) = setParam[Boolean]("nesterov", nesterov)

  //TODO minibatch update
  def get_minibatch(X: DenseMatrix[Double], y: Seq[Double]): Seq[(DenseMatrix[Double], Seq[Double])] = {

    assert(X.rows == y.size)
    if (batchSize >= y.size) Seq((X, y))
    else {
      val (new_X, new_y) = MatrixTools.shuffle(X, y)
      MatrixTools.vsplit(new_X, new_y, batchSize)
    }
  }


  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    var velocity = DenseVector.zeros[Double](x.cols)

    breakable {
      var last_avg_loss = Double.MaxValue
      var last_grad = DenseVector.zeros[Double](x.cols)
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(_weight)

          val y_format = format_y(y(i), loss)

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          doPenalty(penalty, eta, lambda)

          val grad = dloss * ele
          if (nesterov) {
            //nestrov momentum update, origin paper version
            //            velocity  = velocity * gamma + grad + (grad - last_grad) * gamma

            //pytorch version, really fast
            velocity = velocity * gamma + grad
            _weight += -eta * velocity
          } else {
            //momentum update
            velocity = gamma * velocity + grad * eta
            _weight += -velocity
          }

          totalLoss += loss.loss(y_pred, y_format)
          last_grad = grad
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

  override def predict(X: Matrix[Double]): Seq[Double] = {
    assert(weight != null)
    predict_lr(X)
  }

  override def predict_proba(X: Matrix[Double]): Seq[Double] = predict_proba_lr(X)
}
