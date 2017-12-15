package com.lx.algos.utils

import breeze.linalg.DenseVector
import Math.signum

/**
  *
  * @project scalaML
  * @author lx on 12:18 PM 23/11/2017
  */

class WeightVector {

  private val MIN_LR_EPS = 1e-6

  protected var _weight: DenseVector[Double] = null

  def weight_init(n: Int): Unit = {
    if(_weight == null) _weight = DenseVector.rand[Double](n + 1)
  }

  def weight: DenseVector[Double] = {
    if(_weight != null) _weight.slice(1, _weight.length)
    else null
  }

  def intercept: Double = {
    if(_weight != null) _weight(0)
    else 0.0
  }

  // Performs L2 regularization scaling
  def l2penalty(lr: Double, lambda: Double): Unit = {
    if (lr >= MIN_LR_EPS) _weight *= (1 - Math.max(0, lambda * lr))
  }

  def l2penalty(lr: DenseVector[Double], lambda: Double): Unit = {
    _weight -= lambda * lr *:* _weight
  }

  // Performs L1 regularization scaling
  def l1penalty(lr: Double, lambda: Double): Unit = {
    if (lr >= MIN_LR_EPS) {
      val l1_g = _weight.map(signum)
      _weight -= lr * lambda * l1_g
    }
  }

  def l1penalty(lr: DenseVector[Double], lambda: Double): Unit = {
    val l1_g = _weight.map(signum)
    _weight -=  lambda * l1_g *:* lr
  }


  def doPenalty(penalty: String, lr: DenseVector[Double], lambda: Double): Unit = {
    penalty match {
      case "l2" => l2penalty(lr, lambda)
      case "l1" => l1penalty(lr, lambda)
      case _ =>   //donothing
    }
  }

  def doPenalty(penalty: String, lr: Double, lambda: Double): Unit = {
    penalty match {
      case "l2" => l2penalty(lr, lambda)
      case "l1" => l1penalty(lr, lambda)
      case _ =>   //do nothing
    }
  }
}
