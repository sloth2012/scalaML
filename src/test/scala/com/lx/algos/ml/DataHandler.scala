package com.lx.algos.ml

import breeze.linalg.{DenseMatrix, DenseVector}
import com.mattg.util.FileUtil
/**
  *
  * @project scalaML
  * @author lx on 5:57 PM 16/11/2017
  */

object DataHandler {
  val fileUtil = new FileUtil

  def binary_cls_data() = {

    val root = getClass.getResource("/data/ml/binary_classify").getPath
    val f_path = root + "/binary_cls_features"
    val y_path = root + "/binary_cls_label"

    //    println(root)
    val features = fileUtil.readLinesFromFile(f_path).map {
      case line: String =>
        line.split(",").map(_.toDouble)
    }

    val cates = fileUtil.readLinesFromFile(y_path).map(_.toDouble)


    (DenseMatrix(features.toArray: _*), DenseVector(cates.toArray))
  }

  def main(args: Array[String]): Unit = {
    val (x, y) = binary_cls_data()
    println(y.length)
  }
}
