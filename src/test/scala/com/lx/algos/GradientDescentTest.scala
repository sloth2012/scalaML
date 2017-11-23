package com.lx.algos

import breeze.linalg.{DenseMatrix, DenseVector}
import com.lx.algos.loss.LogLoss
import com.lx.algos.optim._
import com.lx.algos.utils.MatrixTools
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 3:50 PM 20/11/2017
  */

class GradientDescentTest extends FlatSpec {

  val (x, y) = DataHandler.binary_cls_data()
  val (new_x, new_y) = MatrixTools.shuffle(x, y.toArray.toSeq)

  val loss = new LogLoss

  {
    val sgd = new SGD
    println(s"this is ${sgd.getClass.getSimpleName} running!")
    sgd.set_verbose(true)
      .set_printPeriod(1)
      .set_eta(0.01)
//            .set_penalty("l1")
          .set_nesterov(true)

    sgd.fit(x, y.toArray.toSeq)
    println(sgd.weight)
  }

  {
    val adagrad = new AdaGrad
    println(s"this is ${adagrad.getClass.getSimpleName} running!")
    adagrad.set_verbose(true)
      .set_printPeriod(10)
      .set_eta(0.01)
//      .set_penalty("l2")

    adagrad.fit(x, y.toArray.toSeq)
    println(adagrad.weight)
  }


  {
    val rmsprop = new RMSProp
    println(s"this is ${rmsprop.getClass.getSimpleName} running!")
    rmsprop.set_verbose(true)
      .set_printPeriod(1)

    rmsprop.fit(x, y.toArray.toSeq)
    println(rmsprop.weight)
  }

  {
    val adadelta = new AdaDelta
    println(s"this is ${adadelta.getClass.getSimpleName} running!")
    adadelta.set_verbose(true)
      .set_printPeriod(1)
      .set_gamma(0.9)
      .fit(x, y.toArray.toSeq)

    println(adadelta.weight)
  }

  {
    val adam = new Adam
    println(s"this is ${adam.getClass.getSimpleName} running!")
    adam.set_verbose(true)
      .set_printPeriod(1)
      .set_gamma(0.9)
      .fit(x, y.toArray.toSeq)

    println(adam.weight)
  }
}
