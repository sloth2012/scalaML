package com.lx.algos.ml.optim.GradientDescent

import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import com.lx.algos.ml._
import com.lx.algos.ml.loss.{LogLoss, LossFunction}
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.norm.{DefaultNormFunction, L1NormFunction, L2NormFunction}
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{MatrixTools, Param, SimpleAutoGrad}

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
      "nesterov" -> false, // NAG支持，改良版，需要配合gamma参数,该部分实现应用了armijo准则
      "verbose" -> false, //打印日志
      "printPeriod" -> 100,
      "penalty" -> "l2", //正则化系数，暂只实现l2
      "iterNum" -> 1000, //迭代轮数
      "loss" -> new LogLoss,
      "batchSize" -> 128 //minibatch size，一个batc多少样本
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
          val y_pred: Double = ele.dot(_theta.toDenseVector)

          val y_format = format_y(y(i), loss)


          val autoGrad = new SimpleAutoGrad(ele, y_format, _theta, loss,  penaltyNorm, lambda)
          val grad = autoGrad.grad

          if (nesterov) {
            //nestrov momentum update, origin paper version
            //            velocity  = velocity * gamma + grad + (grad - last_grad) * gamma

            //pytorch version, really fast
            velocity = velocity * gamma + grad
            _theta += (-eta * velocity).toDenseMatrix.reshape(_theta.rows, 1)
          } else {
            //momentum update
            velocity = gamma * velocity + grad * eta
            _theta += -velocity.toDenseMatrix.reshape(_theta.rows, 1)
          }

          autoGrad.updateTheta(_theta)
          totalLoss += autoGrad.loss
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
