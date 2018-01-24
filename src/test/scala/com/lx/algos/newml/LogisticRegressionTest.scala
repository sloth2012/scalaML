package com.lx.algos.newml

import com.lx.algos.data.DataHandler
import com.lx.algos.newml.metrics.ClassificationMetrics
import com.lx.algos.newml.model.classification.LogisticRegression
import com.lx.algos.newml.norm.L2Norm
import com.lx.algos.newml.optim.GradientDescent._
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 6:13 PM 22/01/2018
  */

class LogisticRegressionTest extends FlatSpec{

  val mnist = DataHandler.mnist_data
  val (x, y) = (mnist.trainDataset.featuresMatrices, mnist.trainDataset.labelMatrices)
  val (xt, yt) = (mnist.testDataset.featuresMatrices, mnist.testDataset.labelMatrices)

  val model = new LogisticRegression

  val solver = new Adam

  model.verbose = true
  model.iterNum = 2000
  model.logPeriod = 1
  model.batchSize = 128
  model.normf = L2Norm
  model.solver = solver

  model.fit(x, y)

  val yt_pred = model.predict(xt)

  val acc = ClassificationMetrics.accuracy_score(yt_pred, yt)
  println(s"test acc is: $acc")
}
