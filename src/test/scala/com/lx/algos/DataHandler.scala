package com.lx.algos

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source

object DataHandler {

  def binary_cls_data() = {

    val root = getClass.getResource("/data").getPath
    val f_path = root + "/cls.feature"
    val y_path = root + "/cls.class"

    //    println(root)
    val features = Source.fromFile(f_path).getLines map {
      case line: String =>
        line.split(",").map(_.toDouble)
    }

    val cates = Source.fromFile(y_path).getLines().map(_.toDouble)


    (DenseMatrix(features.toArray: _*), DenseVector(cates.toArray))
  }

  def main(args: Array[String]): Unit = {

    val (x, y) = binary_cls_data()

    println(y.length)

  }
}
