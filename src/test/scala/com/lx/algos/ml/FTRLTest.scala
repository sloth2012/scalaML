package com.lx.algos.ml

import com.lx.algos.data.DataHandler
import com.lx.algos.ml.loss.LogLoss
import com.lx.algos.ml.optim.GradientDescent.FTRL_Proximal
import com.lx.algos.ml.utils.MatrixTools
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 12:21 AM 24/11/2017
  */

class FTRLTest extends FlatSpec {
  val (x, y) = DataHandler.binary_cls_data()
  val (new_x, new_y) = MatrixTools.shuffle(x, y.toArray.toSeq)

  val loss = new LogLoss

  {
    val model = new FTRL_Proximal(x.cols)
    println(s"this is ${model.getClass.getSimpleName} running!")
    model.set_verbose(true)
      .set_printPeriod(100)
      .set_loss(loss)
      .fit(x, y.toArray.toSeq)

    println(model.weight)
  }
}
