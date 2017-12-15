package com.lx.algos

import breeze.linalg.DenseMatrix
import breeze.numerics.pow
import com.lx.algos.optim.newton.DFP
import com.lx.algos.utils.MatrixTools
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
  model.set_penalty("l2")
      .set_verbose(true)
      .set_lambda(0.1)
      .fit(new_x, new_y)

  println(model.weight)


}
