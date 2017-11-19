package com.lx.algos
/**
  *
  * @project scalaML
  * @author lx on 6:21 PM 16/11/2017
  */

import org.scalatest.FlatSpec
import breeze.linalg._
import com.lx.algos.loss.{HingeLoss, LogLoss, ModifiedHuber}
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.optim.{BASEBGD, BASEMBGD, BASESGD}
import com.lx.algos.utils.MatrixTools

class BreezeTest extends FlatSpec {

  val (x, y) = DataHandler.binary_cls_data()
  val loss = new LogLoss


  val (new_x, new_y) = MatrixTools.shuffle(x, y.toArray.toSeq)


//  {
//    val bgd = new BASEBGD(0.001, 0.15, loss, 10000, verbose = true,  print_period = 500)
//
//    bgd.fit(x, y.toArray.toSeq)
//    val y_pred = bgd.predict(x)
//    println(bgd.weight)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
//  }
//
//  {
//    val sgd = new BASESGD(0.001, 0.15, loss, 10000, verbose = true, print_period = 100)
//
//    sgd.fit(x, y.toArray.toSeq)
//    val y_pred = sgd.predict(x)
//    println(sgd.weight)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
//
//  }

  {
    val mbgd = new BASEMBGD(0.001, 0.15, loss, 10000, verbose = true, batch = 5000)

    mbgd.fit(x, y.toArray.toSeq)
    val y_pred = mbgd.predict(x)
    println(mbgd.weight)
    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
  }



}


