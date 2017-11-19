
package com.lx.algos.optim

/**
  *
  * @project scalaML
  * @author lx on 4:57 PM 16/11/2017
  */

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, norm, shuffle}
import com.lx.algos._
import com.lx.algos.loss.LossFunction
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.utils.{MatrixTools, Param, TimeUtil}

import scala.util.control.Breaks._

trait Optimizer {
  var MIN_LOSS: Double = 1e-6


  var weight: DenseVector[Double] = null
  var intercept: Double = 0

  def fit(X: Matrix[Double], y: Seq[Double]): Optimizer

  def predict(X: Matrix[Double]): Seq[Double]

  def predict_proba(X: Matrix[Double]): Seq[Double]

  // Performs L2 regularization scaling
  def scaleWeights(n: Double): Unit = {
    weight *= n
  }

  def predict_lr(X: Matrix[Double]): Seq[Double] = {
    assert(weight != null)
    (X * weight + intercept).toArray.map {
      case y: Double => if (y > 0) 1.0 else 0
    }
  }

  def predict_proba_lr(X: Matrix[Double]): Seq[Double] = {
    assert(weight != null)

    (X * weight + intercept).toArray.map {
      case y: Double => 1.0 / Math.log(1 + Math.exp(-y))
    }
  }

  protected def log_print(epoch: Int, acc: Double, avg_loss: Double): Unit = {
    println(s"iteration ${epoch + 1}: norm:${norm(weight)}, bias:$intercept, avg_loss:$avg_loss, acc:${acc.formatted("%.6f")}")
  }
}

//批量随机下降
class BASEBGD(var eta: Double, //学习速率
              lambda: Double, // : Double, //正则化参数
              loss: LossFunction, //损失函数
              iterNum: Int = 1000, //迭代次数
              penalty: String = "l2", // 暂时只实现L2正则化
              verbose: Boolean = false,
              print_period: Int = 100 //打印周期
             ) extends Optimizer {

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    val x = X.toDenseMatrix
    weight = DenseVector.zeros[Double](x.cols) //init_weight

    breakable {
      var last_avg_loss = Double.MaxValue
      for (epoch <- 0 until iterNum) {

        var totalLoss: Double = 0

        val delta_w = DenseVector.zeros[Double](x.cols)
        var delta_b: Double = 0.0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(weight) + intercept
          val y_format = if (y(i) == 1.0) 1.0 else -1.0

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          val update = -dloss
          delta_w += ele * update
          delta_b += update

          totalLoss += loss.loss(y_pred, y_format)
        }
        if (penalty == "l2") scaleWeights(Math.max(0, 1 - eta * lambda))

        println()
        weight += delta_w * (1.0 / x.rows) * eta
        intercept += delta_b * (1.0 / x.rows) * eta


        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if ((epoch + 1) % print_period == 0 || epoch == iterNum - 1) {
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
class BASESGD(var eta: Double, //学习速率
              lambda: Double, //正则化参数
              loss: LossFunction, //损失函数
              iterNum: Int = 1000, //迭代次数
              penalty: String = "l2", // 暂时只实现L2正则化
              verbose: Boolean = false,
              print_period: Int = 100 //打印周期
             ) extends Optimizer {

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    val x = X.toDenseMatrix
    weight = DenseVector.rand[Double](x.cols) //init_weight

    breakable {
      var last_avg_loss = Double.MaxValue
      for (epoch <- 0 until iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(weight) + intercept

          val y_format = if (y(i) == 1.0) 1.0 else -1.0 //需要注意，分类损失函数的格式化为-1和1

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss

          val update = -eta * dloss
          if (penalty == "l2") scaleWeights(Math.max(0, 1 - eta * lambda))

          weight += ele * update
          intercept += update

          totalLoss += loss.loss(y_pred, y_format)
        }
        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if ((epoch + 1) % print_period == 0 || epoch == iterNum - 1) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, avg_loss)
          }
        }
        if (converged) {
          println(s"converged at iter ${epoch+1}!")
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
class BASEMBGD(var eta: Double, //学习速率
               lambda: Double, //正则化参数
               loss: LossFunction, //损失函数
               iterNum: Int = 1000, //迭代次数
               penalty: String = "l2", // 暂时只实现L2正则化
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

    val x = X.toDenseMatrix
    weight = DenseVector.rand[Double](x.cols) //init_weight

    breakable {
      var last_avg_loss = Double.MaxValue

      val batch_data = TimeUtil.timer(get_minibatch(x,y,batch), "get_minibatch")
      for (epoch <- 0 until iterNum) {

        var totalLoss: Double = 0

        for ((sub_x, sub_y) <- batch_data.asInstanceOf[Seq[(DenseMatrix[Double], Seq[Double])]]) {
          val delta_w = DenseVector.zeros[Double](x.cols)
          var delta_b: Double = 0.0
          for (i <- 0 until sub_x.rows) {
            val ele = sub_x(i, ::).t
            val y_pred: Double = ele.dot(weight) + intercept

            val y_format = if (sub_y(i) == 1.0) 1.0 else -1.0 //需要注意，分类损失函数的格式化为-1和1

            var dloss = loss.dLoss(y_pred, y_format)

            dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
            else if (dloss > MAX_DLOSS) MAX_DLOSS
            else dloss

            val update = -dloss
            delta_w += ele * update
            delta_b += update

            totalLoss += loss.loss(y_pred, y_format)

          }

          if (penalty == "l2") scaleWeights(Math.max(0, 1 - eta * lambda))

          weight += delta_w * (1.0 / sub_x.rows) * eta
          intercept += delta_b * (1.0 / sub_x.rows) * eta
        }

        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if ((epoch + 1) % print_period == 0 || epoch == iterNum - 1) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, avg_loss)
          }
        }
        if (converged) {
          println(s"converged at iter ${epoch+1}!")
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
class SGD(var eta: Double, //学习速率
          lambda: Double, //正则化参数
          loss: LossFunction, //损失函数
          iterNum: Int = 1000, //迭代次数
          penalty: String = "l2", // 暂时只实现L2正则化
          verbose: Boolean = false,
          print_period: Int = 100 //打印周期
         ) extends Optimizer with Param {


  def get_minibatch(X: DenseMatrix[Double], y: Seq[Double], minibatch_size: Int): Seq[(DenseMatrix[Double], Seq[Double])] = {
    val (new_X, new_y) = MatrixTools.shuffle(X, y)
    MatrixTools.vsplit(new_X, new_y, minibatch_size)
  }


  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {
    assert(X.rows == y.size)

    val x = X.toDenseMatrix
    weight = DenseVector.rand[Double](x.cols) //init_weight

    breakable {
      var last_avg_loss = Double.MaxValue
      for (epoch <- 0 until iterNum) {

        var totalLoss: Double = 0

        for (i <- 0 until x.rows) {
          val ele = x(i, ::).t
          val y_pred: Double = ele.dot(weight) + intercept

          val y_format = if (y(i) == 1.0) 1.0 else -1.0 //需要注意，分类损失函数的格式化为-1和1

          var dloss = loss.dLoss(y_pred, y_format)

          dloss = if (dloss < -MAX_DLOSS) -MAX_DLOSS
          else if (dloss > MAX_DLOSS) MAX_DLOSS
          else dloss


          if (penalty == "l2") scaleWeights(Math.max(0, 1 - eta * lambda))

          val last_w = weight.copy
          //momentum
          //          {
          //            val gamma = getParam[Double]("gamma", 0.9) //一般取0.9
          //
          //            val update = -eta * dloss
          //
          //            val delta_w = gamma * last_w
          //          }

          //TODO delete
          val update = -eta * dloss
          weight += ele * update
          intercept += update

          totalLoss += loss.loss(y_pred, y_format)
        }
        val avg_loss = totalLoss / x.rows

        val converged = Math.abs(avg_loss - last_avg_loss) < MIN_LOSS

        if (verbose) {
          if ((epoch + 1) % print_period == 0 || epoch == iterNum - 1) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, avg_loss)
          }
        }
        if (converged) {
          println(s"converged at iter ${epoch+1}!")
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