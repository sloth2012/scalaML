package com.lx.algos.newml.optim.newton


/**
  *
  * @project scalaML
  * @author lx on 5:30 PM 25/01/2018
  */


sealed trait Interpolation {
  //所有参数顺序为
  def getValue(args: Double*): Double
}

//二分插值，认为函数为线性，每次找中点
class BisectionInterpolation extends Interpolation {
  override def getValue(args: Double*): Double = {
    //需要2个参数
    if (args.size < 2) {
      throw new IllegalArgumentException(
        s"""
           |arguments should be getValue(x1, x2)
           |x1, x2: value x
         """.stripMargin)
    }

    val Seq(x1, x2, _*) = args

    (x1 + x2) / 2

  }
}

class QuadraticInterpolation extends Interpolation {
  override def getValue(args: Double*): Double = {
    //需要5个参数
    if (args.size < 5) {
      throw new IllegalArgumentException(
        s"""
           |arguments should be getValue(x1, f1, x2, f2, f1‘)
           |x1, x2: value x
           |f1, f2: function f
           |f1': f1'(x1)
         """.stripMargin)
    }

    val Seq(x1, f1, x2, f2, f1_1, _*) = args
    assert(x1 != x2, "should satisfy x1 != x2")
    val v = x1 - 0.5 * (x1 - x2) / (1 - (f1 - f2) / ((x1 - x2) * f1_1))

    v
  }
}

object Interpolation {

  lazy val bisection = new BisectionInterpolation
  lazy val quadratic = new QuadraticInterpolation
}
