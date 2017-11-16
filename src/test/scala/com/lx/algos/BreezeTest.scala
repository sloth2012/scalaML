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
import com.lx.algos.optim.{BGD, MBGD, SGD}
import com.lx.algos.utils.MatrixTools

class BreezeTest extends FlatSpec {

  val (x, y) = DataHandler.binary_cls_data()
  val loss = new LogLoss

  {
    val bgd = new BGD(0.001, 0.15, loss, 5000)

    bgd.fit(x, y.toArray.toSeq)
    val y_pred = bgd.predict(x)
    println(bgd.weight)
    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
  }

  {
    val sgd = new SGD(0.001, 0.15, loss, 10000, verbose = true, print_period = 500)

    sgd.fit(x, y.toArray.toSeq)
    val y_pred = sgd.predict(x)
    println(sgd.weight)
    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")

  }

  {
    val mbgd = new MBGD(0.01, 0.20, loss, 1200, verbose = true)

    mbgd.fit(x, y.toArray.toSeq)
    val y_pred = mbgd.predict(x)
    println(mbgd.weight)
    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
  }



}


