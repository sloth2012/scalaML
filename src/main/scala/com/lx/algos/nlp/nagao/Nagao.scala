package com.lx.algos.nlp.nagao

import java.io.FileWriter

import scala.collection.mutable._
import scala.io.Source
import scala.util.{Failure, Sorting, Success, Try}
import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 5:15 PM 19/12/2017
  */


class Nagao(var N: Int = 5) {

  private var leftPTable: Array[String] = Array.empty[String]
  private var rightPTable: Array[String] = Array.empty[String]

  private var leftLTable: Array[Int] = null
  private var rightLTable: Array[Int] = null

  private val wordTFNeighbor: Map[String, TFNeighbor] = Map.empty[String, TFNeighbor]
  private var wordNumber = 0.0

  private final val stopwords = "的很了么呢是嘛个都也比还这于不与才上用就好在和对挺去后没说"

  //co-prefix length of s1 and s2
  private def coPrefixLength(s1: String, s2: String): Int = {
    var coPrefixLen = 0

    breakable {
      0 until Math.min(s1.length, s2.length) foreach {
        i =>
          if (s1(i) == s2(i)) coPrefixLen += 1
          else break
      }
    }

    coPrefixLen
  }

  //add substring of line to pTable
  private def addToPTable(line: String): Unit = { //split line according to consecutive none Chinese character
    val phrases = line.split("[^\u4E00-\u9FA5]+|[" + stopwords + "]")
    for (phrase <- phrases) {
      val reversePhrase = phrase.reverse
      0 until phrase.length foreach {
        i => {
          rightPTable :+= phrase.substring(i)
          leftPTable :+= reversePhrase.substring(i)
        }
      }
      wordNumber += phrase.length
    }
  }

  //count lTable
  private def countLTable(): Unit = {
    Sorting.quickSort(rightPTable)
    rightLTable = new Array[Int](rightPTable.length)

    Sorting.quickSort(leftPTable)
    leftLTable = new Array[Int](leftPTable.length)

    1 until rightPTable.length foreach {
      i => {
        rightLTable(i) = coPrefixLength(rightPTable(i - 1), rightPTable(i))
        leftLTable(i) = coPrefixLength(leftPTable(i - 1), leftPTable(i))
      }
    }

    println("Info: [Nagao Algorithm Step 2]: having sorted PTable and counted left and right LTable")
  }

  //according to pTable and lTable, count statistical result: TF, neighbor distribution
  private def countTFNeighbor(): Unit = { //get TF and right neighbor

    0 until rightPTable.length foreach {
      pIndex => {
        {
          val phrase = rightPTable(pIndex)
          1 + rightLTable(pIndex) to Math.min(N, phrase.length) foreach {
            length => {
              val word = phrase.substring(0, length)
              val tfNeighbor = wordTFNeighbor.getOrElse(word, new TFNeighbor)
              tfNeighbor.incrementTF()
              if (phrase.length > length) {
                tfNeighbor.addToRightNeighbor(phrase(length))
              }
              breakable {
                pIndex + 1 until rightLTable.length foreach {
                  lIndex => {
                    if (rightLTable(lIndex) >= length) {
                      tfNeighbor.incrementTF()
                      val coPhrase = rightPTable(lIndex)
                      if (coPhrase.length > length) {
                        tfNeighbor.addToRightNeighbor(coPhrase(length))
                      }
                    }
                    else break
                  }
                }
              }
              wordTFNeighbor(word) = tfNeighbor
            }
          }
        }

        {
          val phrase = leftPTable(pIndex)
          1 + leftLTable(pIndex) to Math.min(N, phrase.length) foreach {
            length => {
              val word = phrase.substring(0, length).reverse
              val tfNeighbor = wordTFNeighbor.getOrElse(word, new TFNeighbor)
              if (phrase.length > length) {
                tfNeighbor.addToLeftNeighbor(phrase(length))
              }
              breakable {
                pIndex + 1 until leftLTable.length foreach {
                  lIndex => {
                    if (leftLTable(lIndex) >= length) {
                      val coPhrase = leftPTable(lIndex)
                      if (coPhrase.length > length) {
                        tfNeighbor.addToLeftNeighbor(coPhrase(length))
                      }
                    }
                    else break
                  }
                }
              }
              wordTFNeighbor(word) = tfNeighbor
            }
          }
        }
      }
    }
    println("Info: [Nagao Algorithm Step 3]: having counted TF and Neighbor")
  }

  //according to wordTFNeighbor, count MI of word
  private def countMI(word: String): Double = {
    if (word.length <= 1) 0
    else {
      val coProbability = wordTFNeighbor(word).getTF / wordNumber

      var minMI = Double.MaxValue

      1 until word.length foreach {
        pos => {
          val leftPart = word.substring(0, pos)
          val rightPart = word.substring(pos)

          val leftProbability = wordTFNeighbor(leftPart).getTF / wordNumber
          val rightProbability = wordTFNeighbor(rightPart).getTF / wordNumber

          val tmp_mi = coProbability / (leftProbability * rightProbability)

          if (tmp_mi < minMI) minMI = tmp_mi
        }
      }
      minMI
    }
  }

  //save TF, (left and right) neighbor number, neighbor entropy, mutual information
  private def saveTFNeighborInfoMI(outfile: String, stopwordfile: String, threshold: Seq[String]): Unit = {
    //    Try{
    val stopWords = HashSet.empty[String]

    Source.fromFile(stopwordfile).getLines.foreach {
      line => if (line.length > 1) stopWords.add(line)
    }

    val writer = new FileWriter(outfile)

//    println(wordTFNeighbor.keys)
    wordTFNeighbor.foreach {
      case (key, value) => {
        breakable {
          if (key.length <= 1 || stopWords.contains(key)) break


          val tf = value.getTF
          val leftNeighborNumber = value.getLeftNeighborNumber
          val rightNeighborNumber = value.getRightNeighborNumber
          val mi = countMI(key)
          if (tf > threshold(0).toInt &&
            leftNeighborNumber > threshold(1).toInt &&
            rightNeighborNumber > threshold(2).toInt &&
            mi > threshold(3).toInt
          ) {


            val result = Seq(key, tf, leftNeighborNumber, rightNeighborNumber,
              value.getLeftNeighborEntropy, value.getRightNeighborEntropy, mi).mkString(",")

            println(s"result:$result")
            writer.write(result+"\n")
          }
        }
      }
    }
    writer.close()
    //    } match {
    //      case Success(_) => ()
    //      case Failure(e)  => throw new RuntimeException(e.getMessage)
    //    }
    println("Info: [Nagao Algorithm Step 4]: having saved to file")

  }

  def setN(n: Int): Unit = N = n

}

object Nagao {
  def applyNagao(
                  inputs: Seq[String],
                  outfile: String,
                  stopwordfile: String,
                  n: Int = 3,
                  threshold_str: String = "5,3,3,3"
                ): Unit = {
    val nagao = new Nagao(n)

    inputs.par.foreach {
      case inputfile =>
//        Try {
        Source.fromFile(inputfile).getLines.foreach {
          line => nagao.addToPTable(line)
        }
//      } match {
//        case Success(_) => ()
//        case Failure(e) => throw new RuntimeException(e.getMessage)
//      }
    }
    println("Info: [Nagao Algorithm Step 1]: having added all left and right substrings to PTable")

    //step 2: sort PTable and count LTable
    nagao.countLTable()
    //step3: count TF and Neighbor
    nagao.countTFNeighbor()
    //step4: save TF NeighborInfo and MI
    nagao.saveTFNeighborInfoMI(outfile, stopwordfile, threshold_str.split(","))
  }


  def main(args: Array[String]): Unit = {
    val ins = Array("/Users/lx/Documents/projects_related/知识推理/code/workspace/tmp.txt")
    applyNagao(ins, "/Users/lx/Downloads/test1.txt", "/Users/lx/Documents/projects_related/知识推理/code/workspace/data/stop_words.utf8")

  }
}