package com.lx.algos.utils
/**
  *
  * @project scalaML
  * @author lx on 5:37 PM 16/11/2017
  */

import breeze.linalg.{*, DenseMatrix, sum}

import scala.util.Random.{shuffle => sys_shuffle}
import breeze.storage.Zero
import spire.implicits.{cforRange, cforRange2}

import scala.reflect.ClassTag

object MatrixTools {

  //使用breeze默认的vsplit一直出错，这里自己实现了一个，主要是顺序分割
  def vsplit[T: ClassTag](mat: DenseMatrix[T], batch: Int): Seq[DenseMatrix[T]] = {

    val batch_num = if (mat.rows % batch == 0) mat.rows / batch else mat.rows / batch + 1

    0 until batch_num map {
      i => mat(i until Math.min(mat.rows, (i+1)*batch), ::)
    }
  }

  def vsplit[T: ClassTag](mat: DenseMatrix[T], y: Seq[T], batch: Int): Seq[(DenseMatrix[T], Seq[T])] = {

    val batch_num = if (mat.rows % batch == 0) mat.rows / batch else mat.rows / batch + 1

    0 until batch_num map {
      i => val end_index = Math.min(mat.rows, (i+1)*batch)
        (mat(i * batch until end_index,  ::),  y.slice(i, end_index))
    }
  }

  def shuffle[T: ClassTag](X: DenseMatrix[T], y: Seq[T]): (DenseMatrix[T], Seq[T]) = {
    val index = sys_shuffle(0 until y.size toList)


    val values = index flatMap {
      i => X(i, ::).t.toArray
    } toArray

    val new_X: DenseMatrix[T] = new DenseMatrix(X.cols, X.rows, values).t //DenseMatrix是以row为偏移构建矩阵的，因此这里需要从列开始


    val new_y = index map {
      i => y(i)
    }

    (new_X, new_y)
  }

}
