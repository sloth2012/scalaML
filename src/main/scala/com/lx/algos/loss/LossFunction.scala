package com.lx.algos.loss

trait LossFunction {
  def loss(p: Double, y: Double): Double = 0

  //求导
  def dLoss(p: Double, y: Double): Double = 0
}
