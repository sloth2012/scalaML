package com.lx

package object algos {

  val LEARNING_RATE_TYPES = Map("constant" -> 1, "optimal" -> 2, "invscaling" -> 3, "pa1" -> 4, "pa2" -> 5)

  val PENALTY_TYPES = Map("none" -> 0, "l2" -> 2, "l1" -> 1, "elasticnet" -> 3)

  val DEFAULT_EPSILON = 0.1

   val MAX_DLOSS = 1e12
}
