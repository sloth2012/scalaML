package com.lx.algos.ml.loss

/**
  *
  * @project scalaML
  * @author lx on 12:02 PM 29/12/2017
  */

//用于多分类，包括0/1这种二分类
trait MultiClassification extends LossFunction

class MultiLogLoss extends MultiClassification {

}
