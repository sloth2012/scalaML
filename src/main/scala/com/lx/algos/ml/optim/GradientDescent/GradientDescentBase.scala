
package com.lx.algos.ml.optim.GradientDescent

/**
  *
  * @project scalaML
  * @author lx on 4:57 PM 16/11/2017
  */


import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import com.lx.algos.ml.loss.LossFunction
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.norm.{DefaultNormFunction, L1NormFunction, L2NormFunction}
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{AutoGrad, MatrixTools, SimpleAutoGrad}

import scala.util.control.Breaks._

//批量随机下降
class BaseBGD(var eta: Double, //学习速率
              lambda: Double, // : Double, //正则化参数
              loss: LossFunction, //损失函数
              iterNum: Int = 1000, //迭代次数
              penalty: String = "l2",
              verbose: Boolean = false,
              print_period: Int = 100 //打印周期
             ) extends Optimizer {

  def penaltyNorm = penalty.toLowerCase match {
    case "l1" => L1NormFunction
    case "l2" => L2NormFunction
    case _ => DefaultNormFunction
  }

  override def fit(X: Matrix[Double], Y: Seq[Double]): Optimizer = {
    assert(X.rows == Y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y = format_y(DenseMatrix.create(Y.size, 1, Y.toArray), loss)

    val autoGrad = new AutoGrad(x, y, _theta, loss, penaltyNorm, lambda)

    breakable {
      for (epoch <- 1 to iterNum) {

        val last_avg_loss = autoGrad.avgLoss
        _theta -= eta * autoGrad.avgGrad
        autoGrad.updateTheta(_theta)

        val avg_loss = autoGrad.avgLoss

//        println(Math.abs(avg_loss - last_avg_loss))
        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if (epoch % print_period == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), Y)
            log_print(epoch, acc, avg_loss)
          }
        }
        if (converged) {
          println(s"converged at iter $epoch!")
          val acc = ClassificationMetrics.accuracy_score(predict(X), Y)
          log_print(epoch, acc, avg_loss)
          break
        }

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

//随机梯度下降
class BaseSGD(var eta: Double, //学习速率
              lambda: Double, //正则化参数
              loss: LossFunction, //损失函数
              iterNum: Int = 1000, //迭代次数
              penalty: String = "l2",
              verbose: Boolean = false,
              print_period: Int = 100 //打印周期
             ) extends Optimizer {

  def penaltyNorm = penalty.toLowerCase match {
    case "l1" => L1NormFunction
    case "l2" => L2NormFunction
    case _ => DefaultNormFunction
  }

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix

    breakable {
      var last_avg_loss = Double.MaxValue
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(_theta.toDenseVector)

          val y_format = format_y(y(i), loss)

          val autoGrad = new SimpleAutoGrad(ele, y_format, _theta, loss,  penaltyNorm, lambda)
          val grad: DenseVector[Double] = autoGrad.grad

          _theta -= eta * grad.toDenseMatrix.reshape(grad.length, 1)

          autoGrad.updateTheta(_theta)
          totalLoss += autoGrad.loss
        }
        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if (epoch % print_period == 0 || epoch == iterNum) {
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

//mini batch stochastic gradient descent
class BaseMSGD(var eta: Double, //学习速率
               lambda: Double, //正则化参数
               loss: LossFunction, //损失函数
               iterNum: Int = 1000, //迭代次数
               penalty: String = "l2",
               batch: Int = 5000, //一个batch样本数
               verbose: Boolean = false,
               print_period: Int = 100 //打印周期
              ) extends Optimizer {
  def get_minibatch(X: DenseMatrix[Double], y: Seq[Double], minibatch_size: Int = batch): Seq[(DenseMatrix[Double], Seq[Double])] = {

    assert(X.rows == y.size)
    if (minibatch_size >= y.size) Seq((X, y))
    else {
      val (new_X, new_y) = MatrixTools.shuffle(X, y)

      MatrixTools.vsplit(new_X, new_y, minibatch_size)
    }
  }

  def penaltyNorm = penalty.toLowerCase match {
    case "l1" => L1NormFunction
    case "l2" => L2NormFunction
    case _ => DefaultNormFunction
  }

  override def fit(X: Matrix[Double], Y: Seq[Double]): Optimizer = {
    assert(X.rows == Y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y = format_y(DenseMatrix.create(Y.size, 1, Y.toArray), loss)

    breakable {
      var last_avg_loss = Double.MaxValue

      for (epoch <- 1 to iterNum) {

        val batch_data = get_minibatch(x, y.toArray)

        for ((sub_x, sub_y) <- batch_data.asInstanceOf[Seq[(DenseMatrix[Double], Seq[Double])]]) {

          val subAutoGrad = new AutoGrad(sub_x, DenseMatrix(sub_y).reshape(sub_y.length, 1), _theta, loss, penaltyNorm, lambda)
          _theta -= eta * subAutoGrad.avgGrad
          subAutoGrad.updateTheta(_theta)
        }

        val autoGrad = new AutoGrad(x, y, _theta, loss, penaltyNorm, lambda)
        val avg_loss = autoGrad.avgLoss
        if (verbose) {
          if (epoch % print_period == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), Y)
            log_print(epoch, acc, avg_loss)
          }
        }

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS
        if (converged) {
          println(s"converged at iter $epoch!")
          val acc = ClassificationMetrics.accuracy_score(predict(X), Y)
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


