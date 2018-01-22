package com.lx.algos.newml.model.classification

import breeze.linalg.{Axis, DenseMatrix, norm}
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.dataset.Dataset
import com.lx.algos.newml.loss.{ClassificationLoss, SoftmaxLoss}
import com.lx.algos.newml.metrics.ClassificationMetrics
import com.lx.algos.newml.model.Estimator
import com.lx.algos.newml.norm.{L2Norm, NormFunction}
import com.lx.algos.newml.optim.GradientDescent.SGD
import com.lx.algos.newml.optim.{Optimizer, WeightMatrix}

import scala.util.control.Breaks.breakable

/**
  *
  * @project scalaML
  * @author lx on 4:35 PM 12/01/2018
  */

class LogisticRegression extends Estimator[Double] with WeightMatrix {

  var fit_intercept = true //是否求偏置

  var iterNum = 1000 //迭代数量
  var batchSize = 128 //minibatchsize
  var lambda = 0.15 //正则化系数

  var verbose: Boolean = false //是否打印日志
  var logPeriod: Int = 100 //打印周期，只在verbose为true时有效

  var solver: Optimizer = new SGD()
  var normf: NormFunction = L2Norm
  private val lossf: ClassificationLoss = new SoftmaxLoss


  private def logger(epoch: Int, acc: Double, avg_loss: Double): Unit = {

    val weightNorm = norm(weight, Axis._1)
    println(s"iteration $epoch: norm:${weightNorm}, bias:$intercept, avg_loss:$avg_loss, acc:${acc.formatted("%.6f")}")
  }

  override def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Estimator[Double] = {
    assert(X.rows == y.rows)
    weight_init(X.cols, y.cols)
    var (data, label) = if (fit_intercept) (DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X), y)
    else (X, y)

    val dataset = new Dataset(data, label)

    breakable {
      for (epoch <- 1 to iterNum) {
        val batch_data = dataset.get_minibatch(batchSize)
        for ((sub_x, sub_y) <- batch_data) {
          val autoGrad = new AutoGrad(sub_x, sub_y, _theta, lossf, normf, lambda)
          solver.run(autoGrad, epoch)
        }

        if (verbose) {
          if (epoch % logPeriod == 0 || epoch == iterNum) {

            val acc = ClassificationMetrics.accuracy_score(predict(X), label)
            val avg_loss = ClassificationMetrics.log_loss(predict_proba(X), label)

            logger(epoch, acc, avg_loss)
          }
        }
      }
    }

    this
  }

  override def predict(X: DenseMatrix[Double]): DenseMatrix[Double] = lossf.formatted_value(weight, intercept, X)

  override def predict_proba(X: DenseMatrix[Double]): DenseMatrix[Double] = lossf.value(weight, intercept, X)
}
