package com.lx.algos.ml.optim

import breeze.linalg.{DenseMatrix, Matrix, norm}
import com.lx.algos.ml.loss.{BinaryClassification, LogLoss, LossFunction}
import com.lx.algos.ml.utils.MatrixTools

/**
  *
  * @project scalaML
  * @author lx on 1:53 PM 20/11/2017
  */

//TODO 未来将fit方法分离，采用每个优化算法，单独被调用的模式
trait Optimizer extends WeightVector{

  //迭代终止条件
  var MIN_LOSS: Double = 1e-5

  def input(X: Matrix[Double]): Matrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X.toDenseMatrix)

  def fit(X: Matrix[Double], y: Seq[Double]): Optimizer

  def predict(X: Matrix[Double]): Seq[Double]

  def predict_proba(X: Matrix[Double]): Seq[Double]

  //现在为二分类
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
    println(s"iteration $epoch: norm:${norm(weight)}, bias:$intercept, avg_loss:$avg_loss, acc:${acc.formatted("%.6f")}")
  }

  //不同损失函数接受的输入格式不太一样，因此这里需要对默认的分类标签进行转换
  //TODO 适应其它类型的输入，如多分类等
  protected def format_y(y: Double, loss: LossFunction): Double = {
    loss match {
      case _ : BinaryClassification => if (y == 1.0) 1.0 else -1.0  //需要注意，logloss损失函数的输入标签为-1和1
      case _ => y
    }
  }

  //目前实现只能处理-1和1类别的变量，因此这里需要对非此类的进行转换
  protected def format_y(y: DenseMatrix[Double], loss: LossFunction): DenseMatrix[Double] = y.map(format_y(_, loss))
  protected def format_y(y: Seq[Double], loss: LossFunction): Seq[Double] = y.map(format_y(_, loss))


  def get_minibatch(X: DenseMatrix[Double], y: Seq[Double], minibatch_size: Int): Seq[(DenseMatrix[Double], Seq[Double])] = {

    assert(X.rows == y.size)
    if (minibatch_size >= y.size) Seq((X, y))
    else {
      val (new_X, new_y) = MatrixTools.shuffle(X, y)

      MatrixTools.vsplit(new_X, new_y, minibatch_size)
    }
  }

  def get_minibatch(X: DenseMatrix[Double], y: DenseMatrix[Double], minibatch_size: Int): Seq[(DenseMatrix[Double], DenseMatrix[Double])] = {

    assert(X.rows == y.rows)
    if (minibatch_size >= y.rows) Seq((X, y))
    else {
      val (new_X, new_y) = MatrixTools.shuffle(X, y)

      MatrixTools.vsplit(new_X, new_y, minibatch_size)
    }
  }

}
