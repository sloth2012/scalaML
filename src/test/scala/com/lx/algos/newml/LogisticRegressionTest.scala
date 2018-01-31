package com.lx.algos.newml

import breeze.linalg.DenseMatrix
import com.lx.algos.data.DataHandler
import com.lx.algos.newml.metrics.ClassificationMetrics
import com.lx.algos.newml.model.classification.LogisticRegression
import com.lx.algos.newml.model.preprocess.OneHotEncoding
import com.lx.algos.newml.norm.{DefaultNorm, L1Norm, L2Norm}
import com.lx.algos.newml.optim.GradientDescent.{AdaMax, Adam}
import com.lx.algos.newml.optim.newton.{LBFGS, LineSearch}
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 6:13 PM 22/01/2018
  */

class LogisticRegressionTest extends FlatSpec {

  val model = new LogisticRegression
  val solver = new LBFGS
  solver.method = LineSearch.wolfePowell
  model.solver = solver
  model.verbose = true
  model.iterNum = 2000
  model.logPeriod = 1
  model.batchSize = 128
  model.normf = L2Norm
  model.earlyStop = true


    val mnist = DataHandler.mnist_data
    val (x, y) = (mnist.trainDataset.featuresMatrices, mnist.trainDataset.labelMatrices)
    val (xt, yt) = (mnist.testDataset.featuresMatrices, mnist.testDataset.labelMatrices)

    model.fit(x, y)

    val yt_pred = model.predict(xt)

    val acc = ClassificationMetrics.accuracy_score(yt_pred, yt)
    println(s"test acc is: $acc")


  //******************************************
//  val (x, y) = DataHandler.binary_cls_data
//
//  val data = x
//  val ohe = new OneHotEncoding[Double]
//  val label = ohe.fit_transform(y.toDenseMatrix.reshape(y.length, 1))
//
//  model.fit(data, label)


  //********************************************
  //  val a: DenseMatrix[Int] = DenseMatrix.create[Int](3,2, Array(1,2,3,4,5,6))
  //  println(a)
  //  val p = new Param
  //  p.setParam("a", a)
  //  println("***")
  //  val b = p.getParam[DenseMatrix[Int]]("a")
  //  println(b)
}
