package com.lx.algos.newml.optim.newton

/**
  *
  * @project scalaML
  * @author lx on 5:30 PM 25/01/2018
  */

case object Interpolation extends Enumeration {
  type Method = Value
  val BISECTION = Value(0, "二分插值")
  val TWO_POINT_QUADRATIC = Value(1, "二点二次插值")
}
