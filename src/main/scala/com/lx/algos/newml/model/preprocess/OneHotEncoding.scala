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
  val encode: HashMap[T, Double] = HashMap.empty

  def decode: Map[Double, T] = encode.map{
    case (k: T, v: Double) => (v, k)
  }.toMap

  //传入为矩阵，若为一维以上，则将整个向量进行onehotencoding
  override def fit(X: DenseMatrix[T]): Transformer[T, Double] = {
    if(X.cols != 1) throw new RuntimeException("label's shape should be nsamples * 1!")

    0 until X.rows map { i =>
      val ele = X(i, ::).t
      encode += ele(0) -> encode.getOrElse(ele(0), encode.size)
    }

    this
  }

  override def transform(X: DenseMatrix[T]): DenseMatrix[Double] = {
    if(X.cols != 1) throw new RuntimeException("label's shape should be nsamples * 1!")
    val res = DenseMatrix.zeros[Double](X.rows, encode.size)
    0 until X.rows map { i =>
      val ele = X(i, ::).t
      val index: Double = encode.getOrElse(ele(0), -1)

      if (index != -1) {
        res(i, index.toInt) = 1
      }
      else {
        throw new RuntimeException(s"unseen label ${ele(0)} in data!")
      }
    }
    res
  }
}
