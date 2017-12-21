package com.lx.algos.nlp

import java.io.File

import com.lx.algos.nlp.nagao.Nagao
import org.scalatest.FlatSpec

/**
  *
  * @project scalaML
  * @author lx on 5:11 PM 21/12/2017
  */

class NagaoTest extends FlatSpec{

  def subdirs(dir: File): Seq[File] = {
    val d = dir.listFiles.toSeq.filter(_.isDirectory)
    val f = dir.listFiles.toSeq.filter(_.isFile)
    f ++ d.flatMap(subdirs _)
  }

  val inputDir = getClass.getResource("/data/nlp/nagao/news").getPath
  val inputFiles = subdirs(new File(inputDir)).map(_.getAbsolutePath).toArray
  val outFile = getClass.getResource("/data/nlp/nagao/").getPath + "result.txt"
  val stopwordFile = getClass.getResource("/data/nlp/nagao/stop_words.utf8").getPath

//  println(outFile)

  Nagao.applyNagao(inputFiles, outFile, stopwordFile)
}
