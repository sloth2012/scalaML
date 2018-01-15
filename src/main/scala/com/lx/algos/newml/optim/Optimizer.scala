package com.lx.algos.newml.optim

import com.lx.algos.newml.autograd.AutoGrad
import com.lx.algos.newml.utils.Param

/**
  *
  * @project scalaML
  * @author lx on 7:02 PM 11/01/2018
  */

trait Optimizer{
  val variables: Param = null//中间变量参数
  def run(autoGrad: AutoGrad, epoch: Int): Unit //每个batch更新运行
}
