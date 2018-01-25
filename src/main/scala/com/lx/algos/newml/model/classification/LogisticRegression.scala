package com.lx.algos.newml.model.classification

import breeze.linalg.DenseMatrix
import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.dataset.Dataset
import com.lx.algos.newml.loss.{ClassificationLoss, SoftmaxLoss}
import com.lx.algos.newml.metrics.ClassificationMetrics
import com.lx.algos.newml.model.Estimator
import com.lx.algos.newml.norm.{L2Norm, NormFunction}
import com.lx.algos.newml.optim.GradientDescent.SGD
import com.lx.algos.newml.optim.newton.NewtonOptimizer
import com.lx.algos.newml.optim.{EarlyStopping, Optimizer, WeightMatrix}

import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 4:35 PM 12/01/2018
  */

class LogisticRegression extends Estimator[Double] with WeightMatrix {


  var iterNum = 1000 //迭代数量
  var batchSize = 100 //minibatchsize
  var lambda = 0.15 //正则化系数

  var verbose: Boolean = false //是否打印日志
  var logPeriod: Int = 100 //打印周期，只在verbose为true时有效

  var solver: Optimizer = new SGD
  var normf: NormFunction = L2Norm
  val earlyStop: Boolean = true

  private val lossf: ClassificationLoss = new SoftmaxLoss

  private def logger(epoch: Int, acc: Double, avg_loss: Double): Unit = {
    println(s"iteration $epoch: avg_loss:$avg_loss, acc:${acc.formatted("%.6f")}")
  }

  override def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Estimator[Double] = {
    assert(X.rows == y.rows)

    var (data, label) = if (fit_intercept) (DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X), y)
    else (X, y)
    weight_init(data.cols, label.cols)
    val dataset = new Dataset(data, label)

    val earlStopping = new EarlyStopping
    earlStopping.verbose = verbose

    breakable {
      for (epoch <- 1 to iterNum) {
        //主要针对若是采用newton这类二阶方法，不能执行minibatch的情况
        val real_batchSize = if(solver.isInstanceOf[NewtonOptimizer]) data.rows else batchSize
        val batch_data = dataset.get_minibatch(real_batchSize, false)
        for ((sub_x, sub_y) <- batch_data) {
          val autoGrad = new AutoGrad(sub_x, sub_y, _theta, lossf, normf, lambda)
          solver.run(autoGrad, epoch)
          //更新权重
          _theta = solver.variables.getParam[DenseMatrix[Double]]("theta")
        }

        val avg_loss = ClassificationMetrics.log_loss(predict_proba(X), y)
        earlStopping.run(avg_loss, epoch)

        if (verbose) {
          if (epoch % logPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            logger(epoch, acc, avg_loss)
          }
        }

        if(earlStopping.converged){
          break
        }
      }
    }

    this
  }

  override def predict(X: DenseMatrix[Double]): DenseMatrix[Double] = lossf.predict(weight, intercept, X)

  override def predict_proba(X: DenseMatrix[Double]): DenseMatrix[Double] = lossf.predict_proba(weight, intercept, X)
}
