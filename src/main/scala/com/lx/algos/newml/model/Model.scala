package com.lx.algos.newml.model

import breeze.linalg.DenseMatrix

/**
  *
  * @project scalaML
  * @author lx on 4:36 PM 12/01/2018
  */

trait Model

trait Estimator extends Model{

  def fit(X: DenseMatrix[Double], y: DenseMatrix[Double]): Estimator

  def predict(X: DenseMatrix[Double]): DenseMatrix[Double]

  def predict_proba(X: DenseMatrix[Double]): DenseMatrix[Double]
}


trait Transformer extends Model{
  def fit(X: DenseMatrix[Double]): Transformer

  def transform(X: DenseMatrix[Double]): DenseMatrix[Double]


  def fit_transform(X: DenseMatrix[Double]): DenseMatrix[Double] = fit(X).transform(X)
}
