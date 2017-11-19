package com.lx.algos.utils

import scala.collection.mutable
import scala.reflect.ClassTag

/**
  *
  * @project scalaML
  * @author lx on 11:16 AM 19/11/2017
  */

//参数控制
trait Param {

  private val _param: mutable.HashMap[String, Any] = new mutable.HashMap

  def getParam[T: ClassTag](name: String, default: T = null): T = Caster.as[T](_param.getOrElse(name, default))

  def setParam[T: ClassTag](name: String, value: T): Param = {
    _param += (name -> value)
    this
  }
}
