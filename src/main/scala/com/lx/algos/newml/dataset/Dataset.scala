package com.lx.algos.newml.dataset

import breeze.linalg.DenseMatrix

import scala.util.Random.{shuffle => sys_shuffle}

/**
  *
  * @project scalaML
  * @author lx on 10:08 AM 12/01/2018
  */

class Dataset(var data: DenseMatrix[Double], var label: DenseMatrix[Double] = null) {

  def shuffle(inplace: Boolean = false): (DenseMatrix[Double], DenseMatrix[Double]) = {
    if (label != null) {
      assert(data.rows == label.rows)
    }
    val index = sys_shuffle(0 until data.rows toList)
    val values_f = (m: DenseMatrix[Double]) => index flatMap {
      i => m(i, ::).t.toArray
    } toArray

    val new_data = new DenseMatrix(data.cols, data.rows, values_f(data)).t //DenseMatrix是以row为偏移构建矩阵的，因此这里需要从列开始
    val new_label = if (label == null) label else new DenseMatrix(label.cols, label.rows, values_f(label)).t

    if (inplace) {
      data = new_data
      label = new_label
    }

    (data, label)
  }

  def vsplit(batch: Int): Seq[(DenseMatrix[Double], DenseMatrix[Double])] = {

    val batch_num = if (data.rows % batch == 0) data.rows / batch else data.rows / batch + 1
    0 until batch_num map {
      i => {
        val start_index = i * batch
        val end_index = Math.min(data.rows, (i + 1) * batch)

        (data(start_index until end_index, ::), label(start_index until end_index, ::))
      }
    }
  }

  def get_minibatch(minibatch_size: Int, doShffule: Boolean = true): Seq[(DenseMatrix[Double], DenseMatrix[Double])] = {

    if (minibatch_size >= data.rows) Seq((data, label))
    else {
      val (new_data, new_label) = if (doShffule) shuffle()
      else (data, label)

      vsplit(minibatch_size)
    }
  }

}
