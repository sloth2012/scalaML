package com.lx.algos.ml.loss

import breeze.linalg.DenseMatrix
import com.lx.algos.ml.utils.BaseGradFunction

/**
  *
  * @project scalaML
  * @author lx on 1:57 PM 15/11/2017
  */




trait LossFunction extends BaseGradFunction{

  //目前在gradient descent上还未使用
  override def grad(theta: DenseMatrix[Double], x: DenseMatrix[Double]): DenseMatrix[Double] = x

  override def value(theta: DenseMatrix[Double], x: DenseMatrix[Double]) =  x * theta

  def loss(p: Double, y: Double): Double = 0

  //均为n*1的shape
  def loss(p: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(p.rows == y.rows && p.cols == y.cols)

    DenseMatrix.zipMap_d.map(p, y, loss)
  }

  //求导,参照sklearn实现，因此未改名为grad
  def dLoss(p: Double, y: Double): Double = 0

  def dLoss(p: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(p.rows == y.rows && p.cols == y.cols)

    DenseMatrix.zipMap_d.map(p, y, dLoss)
  }
}
