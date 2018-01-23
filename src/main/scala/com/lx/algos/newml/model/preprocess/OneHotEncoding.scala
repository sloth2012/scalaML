package com.lx.algos.newml.model.preprocess

import breeze.linalg.{DenseMatrix, DenseVector}
import com.lx.algos.newml.model.Transformer

import scala.collection.mutable.{HashMap, HashSet}

/**
  *
  * @project scalaML
  * @author lx on 7:12 PM 12/01/2018
  */

class OneHotEncoding[T] extends Transformer[T, Double] {
  val encodeMap: HashMap[Int, Double] = HashMap.empty

  val labels: HashSet[DenseVector[T]] = HashSet.empty

  //传入为矩阵，若为一维以上，则将整个向量进行onehotencoding
  override def fit(X: DenseMatrix[T]): Transformer[T, Double] = {
    0 until X.rows map { i =>
      val ele = X(i, ::).t
      labels.add(ele)
      val key = hashLabel(ele)
      encodeMap += (key -> encodeMap.getOrElse(key, encodeMap.size))
    }

    this
  }

  override def transform(X: DenseMatrix[T]): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](X.rows, encodeMap.size)
    0 until X.rows map { i =>
      val index: Double = encodeMap.getOrElse(hashLabel(X(i, ::).t), -1)

      if (index != -1) {
        res(i, index.toInt) = 1
      }
      else {
        throw new RuntimeException("Unseen Label in data!")
      }
    }

    res
  }


  def hashLabel(data: DenseVector[T]): Int = {
    data.hashCode()
  }

}
