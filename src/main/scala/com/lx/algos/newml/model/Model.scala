package com.lx.algos.newml.model

import breeze.linalg.DenseMatrix

/**
  *
  * @project scalaML
  * @author lx on 4:36 PM 12/01/2018
  */

trait Model

//均需做onehotencoding
trait Estimator[T] extends Model {

  def fit(X: DenseMatrix[T], y: DenseMatrix[T]): Estimator[T]

  def predict(X: DenseMatrix[T]): DenseMatrix[T]

  def predict_proba(X: DenseMatrix[T]): DenseMatrix[T]
}


trait Transformer[A, B] extends Model {
  def fit(X: DenseMatrix[A]): Transformer[A, B]

  def transform(X: DenseMatrix[A]): DenseMatrix[B]


  def fit_transform(X: DenseMatrix[A]): DenseMatrix[B] = fit(X).transform(X)
}
