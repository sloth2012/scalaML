package com.lx.algos.loss
/**
  *
  * @project scalaML
  * @author lx on 1:57 PM 15/11/2017
  */

trait LossFunction {
  def loss(p: Double, y: Double): Double = 0

  //求导
  def dLoss(p: Double, y: Double): Double = 0
}
