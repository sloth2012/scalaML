package com.lx.algos.newml.utils

import breeze.linalg.{DenseMatrix, Matrix}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.collection.immutable

/**
  *
  * @project scalaML
  * @author lx on 11:16 AM 19/11/2017
  */


class Default[+A](val default: A)

trait LowerPriorityImplicits {
  // Stop AnyRefs from clashing with AnyVals
  implicit def defaultNull[A <: AnyRef]:Default[A] = new Default[A](null.asInstanceOf[A])
}

object Default extends LowerPriorityImplicits {
  implicit object DefaultDouble extends Default[Double](0.0)
  implicit object DefaultFloat extends Default[Float](0.0F)
  implicit object DefaultInt extends Default[Int](0)
  implicit object DefaultLong extends Default[Long](0L)
  implicit object DefaultShort extends Default[Short](0)
  implicit object DefaultByte extends Default[Byte](0)
  implicit object DefaultChar extends Default[Char]('\u0000')
  implicit object DefaultBoolean extends Default[Boolean](false)
  implicit object DefaultUnit extends Default[Unit](())

  implicit def defaultSeq[A]: Default[immutable.Seq[A]] = new Default[immutable.Seq[A]](immutable.Seq())
  implicit def defaultSet[A]: Default[Set[A]] = new Default[Set[A]](Set())
  implicit def defaultMap[A, B]: Default[Map[A, B]] = new Default[Map[A, B]](Map[A, B]())
  implicit def defaultOption[A]: Default[Option[A]] = new Default[Option[A]](None)

  def value[A](implicit value: Default[A]): A = value.default
}


//参数控制
class Param {
  private val _param: mutable.HashMap[String, Any] = new mutable.HashMap

  def getParam[T: ClassTag](name: String, default: T = Default.value[T]): T = Caster.as[T](_param.getOrElse(name, default))

  def setParam[T: ClassTag](name: String, value: T): Param = {
    _param += (name -> value)
    this
  }

  def setParams[T <: Iterable[(String, Any)]](params : T): Param = {
    _param ++= params
    this
  }

  def getParam = _param
}


