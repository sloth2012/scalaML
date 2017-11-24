package com.lx.algos

import com.lx.algos.loss.LogLoss
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.optim.BaseMSGD
import com.lx.algos.utils.MatrixTools
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 6:21 PM 16/11/2017
  */


class BaseGradientDescentTest extends FlatSpec {

  val (x, y) = DataHandler.binary_cls_data()
  val loss = new LogLoss


  val (new_x, new_y) = MatrixTools.shuffle(x, y.toArray.toSeq)


//  {
//    val bgd = new BaseBGD(0.001, 0.15, loss, 10000, verbose = true,  print_period = 500)
//
//    bgd.fit(x, y.toArray.toSeq)
//    val y_pred = bgd.predict(x)
//    println(bgd.weight)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
//  }
//
//  {
//    val sgd = new BaseSGD(0.001, 0.15, loss, 10000, verbose = true, print_period = 100)
//
//    sgd.fit(x, y.toArray.toSeq)
//    val y_pred = sgd.predict(x)
//    println(sgd.weight)
//    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
//
//  }

  {
    val mbgd = new BaseMSGD(0.001, 0.15, loss, 10000, verbose = true, batch = 500)

    mbgd.fit(x, y.toArray.toSeq)
    val y_pred = mbgd.predict(x)
    println(mbgd.weight)
    println(s"acc: ${ClassificationMetrics.accuracy_score(y_pred, y.toArray.toSeq)}")
  }


}


