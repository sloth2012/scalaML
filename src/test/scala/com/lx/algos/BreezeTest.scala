package com.lx.algos

import org.scalatest.FlatSpec
import breeze.linalg._
import com.lx.algos.loss.{HingeLoss, LogLoss, ModifiedHuber}
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.optim.{BGD, SGD}

class BreezeTest extends FlatSpec {
  val m1 = DenseMatrix((1, 2, 3), (4, 5, 6))
  val m2 = DenseMatrix((6, 5, 4), (3, 2, 1))

  val w = DenseVector(1, 2, 3)

  val a  = m2(1, ::)

  "matrix add" should "equal pass" in {
    assert(m1 + m2 equals DenseMatrix((7, 7, 7), (7, 7, 7)))
  }

  println(vsplit(m1, 2))

  val (x, y) = DataHandler.binary_cls_data()
  val loss = new LogLoss

//  {
//    val bgd = new BGD(0.001, 0.15, loss, 5000)
//
//    bgd.fit(x, y.toArray.toSeq)
//    val res = bgd.predict(x)
//    println(bgd.weight)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(res, y.toArray.toSeq)}")
//  }

//  {
//    val sgd = new SGD(0.001, 0.15, loss, 10000, verbose = true, print_period = 500)
//
//    sgd.fit(x, y.toArray.toSeq)
//    val res2 = sgd.predict(x)
//    println(sgd.weight)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(res2, y.toArray.toSeq)}")
//
//  }

//  {
//    val sgd = new SGD(0.01, 0.20, loss, 1200)
//
//    sgd.fit(x, y.toArray.toSeq)
//    val res2 = sgd.predict(x)
//    println(sgd.weight)
//    println(sgd.intercept)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(res2, y.toArray.toSeq)}")
//  }



}


