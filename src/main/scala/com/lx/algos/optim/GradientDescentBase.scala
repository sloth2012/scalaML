
package com.lx.algos.optim

/**
  *
  * @project scalaML
  * @author lx on 4:57 PM 16/11/2017
  */

import breeze.linalg.{DenseMatrix, DenseVector, Matrix}
import com.lx.algos._
import com.lx.algos.loss.LossFunction
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.utils.MatrixTools

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

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix

    breakable {
      var last_avg_loss = Double.MaxValue
      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0

        val delta_w = DenseVector.zeros[Double](x.cols)
        var delta_b: Double = 0.0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(_weight)
          val y_format = format_y(y(i), loss)

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          val update = -dloss
          delta_w += ele * update
          delta_b += update

          totalLoss += loss.loss(y_pred, y_format)
        }

        doPenalty(penalty, eta, lambda)
        _weight += delta_w * (1.0 / x.rows) * eta


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

//随机梯度下降
class BaseSGD(var eta: Double, //学习速率
              lambda: Double, //正则化参数
              loss: LossFunction, //损失函数
              iterNum: Int = 1000, //迭代次数
              penalty: String = "l2",
              verbose: Boolean = false,
              print_period: Int = 100 //打印周期
             ) extends Optimizer {

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
          val y_pred: Double = ele.dot(_weight)

          val y_format = format_y(y(i), loss)

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          val update = -eta * dloss
          doPenalty(penalty, eta, lambda)

          _weight += ele * update

          totalLoss += loss.loss(y_pred, y_format)
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
               batch: Int = 128, //一个batch样本数
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

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix

    breakable {
      var last_avg_loss = Double.MaxValue

      for (epoch <- 1 to iterNum) {

        var totalLoss: Double = 0
        val batch_data = get_minibatch(x, y, batch)
        for ((sub_x, sub_y) <- batch_data.asInstanceOf[Seq[(DenseMatrix[Double], Seq[Double])]]) {
          val delta_w = DenseVector.zeros[Double](x.cols)
          var delta_b: Double = 0.0
          for (i <- 0 until sub_x.rows) {
            val ele = sub_x(i, ::).t
            val y_pred: Double = ele.dot(_weight)

            val y_format = format_y(y(i), loss)

            var dloss = loss.dLoss(y_pred, y_format)

            dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
            else if (dloss > MAX_DLOSS) MAX_DLOSS
            else dloss

            val update = -dloss
            delta_w += ele * update
            delta_b += update

            totalLoss += loss.loss(y_pred, y_format)
          }

          doPenalty(penalty, eta, lambda)

          _weight += delta_w * (1.0 / sub_x.rows) * eta
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


