package com.lx.algos.ml

import com.lx.algos.data.DataHandler
import com.lx.algos.ml.loss.{HingeLoss, SquaredHingeLoss}
import com.lx.algos.ml.optim.newton.{BFGS, CG, DFP, LBFGS}
import com.lx.algos.ml.utils.MatrixTools
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 11:02 AM 14/12/2017
  */

class NewtonTest extends FlatSpec {

  val (x, y) = DataHandler.binary_cls_data
  val (new_x, new_y) = MatrixTools.shuffle(x, y.toArray.toSeq)

  {
    val model = new DFP
    println(s"this is ${model.getClass.getSimpleName} running!")
    model.set_penalty("l2")
      .set_verbose(true)
      .set_lambda(0.1)
      .set_printPeriod(1)
      .set_loss(new SquaredHingeLoss)
      .fit(new_x, new_y)

    println(model.weight)
  }

  //
  {
    val model = new BFGS
    println(s"this is ${model.getClass.getSimpleName} running!")
    model.set_penalty("l2")
      .set_verbose(true)
      .set_printPeriod(1)
      .set_lambda(0.1)
      .set_loss(new HingeLoss)
      .fit(new_x, new_y)

    println(model.weight)
  }

  {
    val model = new CG
    println(s"this is ${model.getClass.getSimpleName} running!")
    model.set_penalty("l2")
      .set_verbose(true)
      .set_printPeriod(1)
      .set_lambda(0.1)
      .set_loss(new HingeLoss)
      .fit(new_x, new_y)

    println(model.weight)
  }

  {
    val model = new LBFGS
    println(s"this is ${model.getClass.getSimpleName} running!")
    model.set_m(15)
      .set_penalty("l2")
      .set_verbose(true)
      .set_printPeriod(1)
      .set_lambda(0.1)
      .set_loss(new HingeLoss)
    //      .fit(new_x, new_y)

    model.fit_Wolfe_Powell(new_x, new_y) //测试Wolfe_Powell一维搜索方法

    println(model.weight)
  }


}
