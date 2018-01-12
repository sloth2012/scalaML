package com.lx.algos.newml.optim

import com.lx.algos.ml.utils.AutoGrad
import com.lx.algos.newml.utils.Param

/**
  *
  * @project scalaML
  * @author lx on 7:02 PM 11/01/2018
  */

trait Optimizer{
  val params: Param  = null//初始化参数
  val variables: Param = null//中间变量参数

  var CONVERAGE_EPS: Double = 1e-5 //收敛约束

  def run(autoGrad: AutoGrad): Double //每个batch更新运行
  var counter: Int = 0 //计数器
}
