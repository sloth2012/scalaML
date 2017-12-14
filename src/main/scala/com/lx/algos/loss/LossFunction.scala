package com.lx.algos.loss

import breeze.generic.{MappingUFunc, UFunc}
import breeze.linalg.DenseMatrix

/**
  *
  * @project scalaML
  * @author lx on 1:57 PM 15/11/2017
  */




trait LossFunction {



  def loss(p: Double, y: Double): Double = 0

  def loss(p: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = LossFunc(p, y)

  //求导
  def dLoss(p: Double, y: Double): Double = 0

  def dLoss(p: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = DlossFunc(p, y)

  //用于矩阵批量计算
  private object LossFunc extends UFunc with MappingUFunc{
    implicit object lossImpl2 extends Impl2[Double, Double, Double] {
      override def apply(v1: Double, v2: Double): Double = loss(v1, v2)
    }
  }

  private object DlossFunc extends UFunc with MappingUFunc{
    implicit object dlossImpl2 extends Impl2[Double, Double, Double] {
      override def apply(v1: Double, v2: Double): Double = loss(v1, v2)
    }
  }
}
