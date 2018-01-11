package com.lx.algos.newml.utils

import scala.reflect.ClassTag

/**
  *
  * @project scalaML
  * @author lx on 11:36 AM 19/11/2017
  */

object Caster {
  def as[T: ClassTag](x: Any): T = {
    opt[T](x) match {
      case Some(e) => e
      case None => throw new Exception(s"Data cast failed: item: ${x} => ${implicitly[ClassTag[T]].runtimeClass.getName}")
    }
  }

  def isInstance[T: ClassTag](x: Any): Boolean = opt[T](x) match {
    case Some(_) => true
    case None => false
  }

  /**
    * None if cast fail
    * @param x
    * @tparam T
    * @return
    */
  def opt[T: ClassTag](x: Any): Option[T] = {
    x match {
      case x: T => Some(x.asInstanceOf[T])
      case _ => None
    }
  }

  /**
    * None if input is null
    * DataCastException if cast fail
    * @param x
    * @tparam T
    * @return
    */
  def asOpt[T: ClassTag](x: Any): Option[T] = {
    if (x == null) None
    else Some(as[T](x))
  }
}

