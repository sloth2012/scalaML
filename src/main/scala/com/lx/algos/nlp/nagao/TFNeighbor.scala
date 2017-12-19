package com.lx.algos.nlp.nagao

import scala.collection.mutable.{HashMap, Map}

/**
  *
  * @project scalaML
  * @author lx on 4:52 PM 19/12/2017
  */

class TFNeighbor private[nagao]() {
  private val leftNeighbor:  Map[Char, Int] = new HashMap[Char, Int]().withDefaultValue (0)
  private val rightNeighbor: Map[Char, Int] = new HashMap[Char, Int]().withDefaultValue (0)
  private var tf = 0

  //add word to leftNeighbor
  def addToLeftNeighbor(word: Char): Unit = { //leftNeighbor.put(word, 1 + leftNeighbor.getOrDefault(word, 0));
    leftNeighbor(word) += 1
  }

  //add word to rightNeighbor
  def addToRightNeighbor(word: Char): Unit = { //rightNeighbor.put(word, 1 + rightNeighbor.getOrDefault(word, 0));
    rightNeighbor(word) += 1
  }

  //increment tf
  def incrementTF(): Unit = {
    tf += 1
  }

  def getLeftNeighborNumber: Int = leftNeighbor.size

  def getRightNeighborNumber: Int = rightNeighbor.size

  def getLeftNeighborEntropy: Double = {
    var entropy = 0.0
    var sum = 0.0
    for (number <- leftNeighbor.values) {
      entropy += number * Math.log(number)
      sum += number
    }
    if (sum == 0)  0.0
    else Math.log(sum) - entropy / sum
  }

  def getRightNeighborEntropy: Double = {
    var entropy = 0.0
    var sum = 0.0
    for (number <- rightNeighbor.values) {
      entropy += number * Math.log(number)
      sum += number
    }
    if (sum == 0) 0.0
    else Math.log(sum) - entropy / sum
  }

  def getTF: Int = tf
}
