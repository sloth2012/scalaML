package com.lx.algos.newml.utils

import java.sql.Timestamp
import java.util.Date

import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormat

/**
  *
  * @project scalaML
  * @author lx on 5:56 PM 19/11/2017
  */

object TimeUtil {
  val datetimeFmts = "yyyy-MM-dd HH:mm:ss"
  val dateFmts = "yyyy-MM-dd"

  val datetimeFmt = DateTimeFormat.forPattern(datetimeFmts)
  val dateFmt = DateTimeFormat.forPattern(dateFmts)

  def encodeDate(d: Date): String = dateFmt.print(d.getTime)
  def decodeDate(s: String): Date = new Date(dateFmt.parseDateTime(s).getMillis)

  def encodeDateTime(d: Date): String = datetimeFmt.print(d.getTime)
  def decodeDateTime(s: String): Timestamp =  new Timestamp(datetimeFmt.parseDateTime(s).getMillis)

  def getCurrentTime():String = encodeDateTime(new Date())

  def currentMillis() = DateTime.now().getMillis

  def currentSeconds() = currentMillis() / 1000

  def autoSecondsToTimeString(seconds: Long): String = {
    val hours = seconds / 3600
    val mins = seconds / 60 - hours * 60
    val secs = seconds - hours * 3600 - mins * 60

    s"${
      hours match {
        case 0 => ""
        case x => s"${x}h"
      }
    }${
      mins match {
        case 0 => ""
        case x => s"${x}m"
      }
    }${
      secs match {
        case 0 => ""
        case x => s"${x}s"
      }
    }"
  }

  def timer(func: => Any, name: String = "Func"): Any = {
    val startTime = currentSeconds()

    val res = func

    val endTime = currentSeconds()

    println(s"$name cost: ${endTime - startTime}s")

    res
  }
}
