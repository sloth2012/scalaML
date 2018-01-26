package com.lx.algos.newml.utils

/**
  *
  * @project scalaML
  * @author lx on 6:27 PM 25/01/2018
  */

object Logger {

  var counter = 0

  def printOnce(ele: Any): Unit = {
    printNTimes(ele, 1)
  }

  def printNTimes(ele: Any, n: Int = 100): Unit = {
    if(counter < n) println(ele)
    counter += 1
  }
}
