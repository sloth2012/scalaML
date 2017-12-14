package com.lx.algos

import breeze.linalg.DenseMatrix
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

  val dfp = new DFP
  dfp.fit(new_x, new_y)

}
