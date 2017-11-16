package com.lx.algos.utils

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import spire.implicits.{cforRange, cforRange2}

import scala.reflect.ClassTag

object MatrixTools {

  //使用breeze默认的vsplit一直出差，这里自己实现了一个，主要是顺序分割
  def vsplit[T: ClassTag](mat: DenseMatrix[T], batch: Int): Seq[DenseMatrix[T]] = {

    val batch_num = if (mat.rows % batch == 0) mat.rows / batch else mat.rows / batch + 1

    0 until batch_num map {
      i => mat(i until Math.min(mat.rows, (i+1)*batch), ::)
    }
  }

}
