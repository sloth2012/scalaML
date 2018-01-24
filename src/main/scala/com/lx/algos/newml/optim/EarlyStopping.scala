package com.lx.algos.newml.optim

import breeze.numerics.abs

/**
  *
  * @project scalaML
  * @author lx on 3:57 PM 24/01/2018
  */

class EarlyStopping {

  var maxConvergedNum = 10 //收敛次数
  var eps: Double = 1e-5 //收敛阈值
  var counter = 0 //收敛计数
  var verbose = true

  private var last_metrics: Double = 0 //记录上次的校准指标

  def converged = counter > maxConvergedNum

  //更新参数
  def run(new_metrics: Double, epoch: Int) = {
    if (abs(new_metrics - last_metrics) < eps) {
      counter += 1
      if (verbose && converged) {
        println(s"model is converged in epoch $epoch!")
      }
    }else{
      counter = 0
    }
    last_metrics = new_metrics
  }

}
