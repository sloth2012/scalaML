package com.lx.ml.algos

import com.lx.ml.algos.optim.newton.DFP
import com.lx.ml.algos.utils.MatrixTools
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 11:02 AM 14/12/2017
  */

class NewtonTest  extends FlatSpec{

  val (x, y) = DataHandler.binary_cls_data()
  val (new_x, new_y) = MatrixTools.shuffle(x, y.toArray.toSeq)

  val model = new DFP
  model.set_penalty("l1")
      .set_verbose(true)
      .set_lambda(0.1)
      .fit(new_x, new_y)

  println(model.weight)


}
