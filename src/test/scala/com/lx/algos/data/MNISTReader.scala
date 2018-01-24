package com.lx.algos.data

import java.io.{DataInputStream, File, FileInputStream, FileOutputStream}
import java.net.URL
import java.nio.channels.Channels
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  *
  * @project scalaML
  * @author lx on 8:46 PM 23/01/2018
  */

class MnistFileReader(location: String, fileName: String) {

  private[this] val path = Paths.get(location, fileName)

  if (!Files.exists(path))
    download

  protected[this] val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path.toString)))

  private def download: Unit = {
    val rbc = Channels.newChannel(new URL(s"http://yann.lecun.com/exdb/mnist/$fileName").openStream())
    val fos = new FileOutputStream(s"$location/$fileName")
    fos.getChannel.transferFrom(rbc, 0, Long.MaxValue)
  }

}

class MnistLabelReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()
  val labelNum = 10

  val labelsAsInts = readLabels(0)
  val labelsAsVectors = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](labelNum) { i => if (i == label) 1.0 else 0.0 }
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(stream.readByte(), readLabels(ind + 1))

}

class MnistImageReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val width = stream.readInt()
  val height = stream.readInt()

  val imagesAsMatrices = readImages(0)
  val imagesAsVectors = imagesAsMatrices map { image =>
    DenseVector.tabulate(width * height) { i => image(i / width, i % height) / 255.0 }
  }

  private[this] def readImages(ind: Int): Stream[DenseMatrix[Int]] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(readImage(), readImages(ind + 1))

  private[this] def readImage(): DenseMatrix[Int] = {
    val m = DenseMatrix.zeros[Int](height, width)

    for (y <- 0 until height; x <- 0 until width)
      m(y, x) = stream.readUnsignedByte()

    m
  }

}

/**
  * http://yann.lecun.com/exdb/mnist/
  */
class MnistDataset(location: String, dataset: String) {

  lazy val imageReader = new MnistImageReader(location, s"$dataset-images-idx3-ubyte.gz")
  lazy val labelReader = new MnistLabelReader(location, s"$dataset-labels-idx1-ubyte.gz")

  def imageWidth = imageReader.width
  def imageHeight = imageReader.height

  def labelNum = labelReader.labelNum

  def imagesAsMatrices = imageReader.imagesAsMatrices
  def imagesAsVectors = imageReader.imagesAsVectors

  def labelsAsInts = labelReader.labelsAsInts
  def labelsAsVectors = labelReader.labelsAsVectors

  def examples = imagesAsVectors zip labelsAsVectors

  private[this] def dataFormatter(mat: Stream[DenseVector[Double]], ncols: Int): DenseMatrix[Double] ={

    val nrows = mat.length
    val data = mat.toArray.flatMap(x => x.data)

    DenseMatrix.create(ncols, nrows, data).t
  }

  def featuresMatrices = dataFormatter(imagesAsVectors, imageWidth * imageHeight)

  def labelMatrices = dataFormatter(labelsAsVectors, labelNum)

}

object Mnist {

  val location =  getClass.getResource("/data/mnist").getPath
  val locationFile = new File(location)

  if (!locationFile.exists)
    locationFile.mkdirs

  val trainDataset = new MnistDataset(location, "train")
  val testDataset = new MnistDataset(location, "t10k")

}
