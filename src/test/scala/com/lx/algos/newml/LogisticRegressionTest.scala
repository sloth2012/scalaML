package com.lx.algos.newml

import com.lx.algos.data.DataHandler
import com.lx.algos.newml.metrics.ClassificationMetrics
import com.lx.algos.newml.model.classification.LogisticRegression
import com.lx.algos.newml.model.preprocess.OneHotEncoding
import com.lx.algos.newml.norm.{DefaultNorm, L1Norm, L2Norm}
import com.lx.algos.newml.optim.GradientDescent.SGD
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

  val solver = new SGD
  solver.nestrov = false
  solver.lr = 0.01

  model.verbose = true
  model.iterNum = 2000
  model.logPeriod = 1
  model.batchSize = 128
  model.normf = L2Norm
  model.solver = solver

  model.fit(data, label)

}
