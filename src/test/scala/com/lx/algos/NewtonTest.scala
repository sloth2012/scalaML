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

  val dfp = new DFP
  dfp.fit(new_x, new_y)


//  val a = DenseMatrix.create(1,4, Array(1.0,2,3,4))
//
//  val b = DenseMatrix.create(1,4, Array(4,2.0,3,43))
//
//  val f = (aa: Double, bb: Double) => aa - bb
//
//  println(DenseMatrix.zipMap_d.map(a,b,f))


}
