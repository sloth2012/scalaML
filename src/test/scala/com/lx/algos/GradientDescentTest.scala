package com.lx.algos

import com.lx.algos.loss.LogLoss
import com.lx.algos.optim.SGD
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
    sgd.set_verbose(true)
        .set_printPeriod(1)
    sgd.fit(x, y.toArray.toSeq)

  }

}
