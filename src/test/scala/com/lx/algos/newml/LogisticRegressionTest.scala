package com.lx.algos.newml

import com.lx.algos.data.DataHandler
import com.lx.algos.newml.model.classification.LogisticRegression
import com.lx.algos.newml.model.preprocess.OneHotEncoding
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 6:13 PM 22/01/2018
  */

class LogisticRegressionTest extends FlatSpec{
  val (x, y) = DataHandler.binary_cls_data()

  val data = x
  val new_y = y.toDenseMatrix.reshape(x.rows, 1)

  val ohe  = new OneHotEncoding[Double]
  ohe.fit(new_y)

  val label = ohe.transform(new_y)
  val model = new LogisticRegression

  model.verbose = true

  model.fit(data, label)

}
