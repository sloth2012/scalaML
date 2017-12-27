package com.lx.algos.ml.utils

/**
  *
  * @project scalaML
  * @author lx on 5:37 PM 16/11/2017
  */

import breeze.linalg.DenseMatrix

import scala.reflect.ClassTag
import scala.util.Random.{shuffle => sys_shuffle}

object MatrixTools {

  //使用breeze默认的vsplit一直出错，这里自己实现了一个，主要是顺序分割
  def vsplit[T: ClassTag](mat: DenseMatrix[T], batch: Int): Seq[DenseMatrix[T]] = {

    val batch_num = if (mat.rows % batch == 0) mat.rows / batch else mat.rows / batch + 1

    0 until batch_num map {
      i => mat(i * batch until Math.min(mat.rows, (i+1)*batch), ::)
    }
  }

  def vsplit[T: ClassTag](mat: DenseMatrix[T], y: Seq[T], batch: Int): Seq[(DenseMatrix[T], Seq[T])] = {

    val batch_num = if (mat.rows % batch == 0) mat.rows / batch else mat.rows / batch + 1

    0 until batch_num map {
      i => {
        val start_index = i * batch
        val end_index = Math.min(mat.rows, (i + 1) * batch)

        (mat(start_index until end_index, ::), y.slice(start_index, end_index))
      }
    }
  }

  def vsplit[T: ClassTag](X: DenseMatrix[T], y: DenseMatrix[T], batch: Int): Seq[(DenseMatrix[T], DenseMatrix[T])] = {

    val batch_num = if (X.rows % batch == 0) X.rows / batch else X.rows / batch + 1

    0 until batch_num map {
      i => {
        val start_index = i * batch
        val end_index = Math.min(X.rows, (i + 1) * batch)

        (X(start_index until end_index, ::), y(start_index until end_index, ::))
      }
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


  def shuffle[T: ClassTag](X: DenseMatrix[T], y: DenseMatrix[T]): (DenseMatrix[T], DenseMatrix[T]) = {
    val index = sys_shuffle(0 until y.size toList)


    val values_f = (m: DenseMatrix[T]) => index flatMap {
      i => m(i, ::).t.toArray
    } toArray

    val new_X: DenseMatrix[T] = new DenseMatrix(X.cols, X.rows, values_f(X)).t //DenseMatrix是以row为偏移构建矩阵的，因此这里需要从列开始


    val new_y: DenseMatrix[T] = new DenseMatrix(y.size, 1, values_f(y))

    (new_X, new_y)
  }

  def shuffle[T: ClassTag](X: DenseMatrix[T]): DenseMatrix[T] = {
    val index = sys_shuffle(0 until X.rows toList)


    val values = index flatMap {
      i => X(i, ::).t.toArray
    } toArray

    val new_X: DenseMatrix[T] = new DenseMatrix(X.cols, X.rows, values).t //DenseMatrix是以row为偏移构建矩阵的，因此这里需要从列开始


    new_X
  }
}
