package com.lx.algos.newml.optim.newton

import breeze.linalg.{DenseMatrix, min, sum}
import breeze.numerics.abs
import com.lx.algos.newml.autograd.AutoGrad

import scala.collection.mutable.ArrayBuffer

/**
  *
  * @project scalaML
  * @author lx on 10:38 AM 25/01/2018
  */


class LBFGS(var m: Int = 8, //保存最近的m次信息，sk和yk，一般在3到20之间
            var method: LineSearch = LineSearch.goldSection, //一维搜索方法
            var tolerance_grad: Double = 1e-5 //termination tolerance on first order optimality
           ) extends NewtonOptimizer {


  override def run(autoGrad: AutoGrad, epoch: Int): Unit = {
    val grad = autoGrad.grad
    var theta = variables.getParam[DenseMatrix[Double]]("theta", autoGrad.theta)

    //校正矩阵
    var dk = variables.getParam[DenseMatrix[Double]]("dk", -autoGrad.grad)

    if (sum(abs(dk)) > tolerance_grad) {

      //表示yk和sk的起始点，即第一个元素的位置，用以每次只抛弃修改一个位置
      var pos = variables.getParam[Int]("pos", -1)
      //保存的m个yk
      var yk_seq = variables.getParam[ArrayBuffer[DenseMatrix[Double]]]("yk_seq", ArrayBuffer.empty[DenseMatrix[Double]])
      //保存的m个sk
      var sk_seq = variables.getParam[ArrayBuffer[DenseMatrix[Double]]]("sk_seq", ArrayBuffer.empty[DenseMatrix[Double]])

      //############################################
      //compute step length
      //############################################

      val alpha = method.getStep(autoGrad, dk)

      //##############################################
      //update dk, yk_seq, sk_seq, theta st.
      //##############################################
      var sk = alpha * dk
      theta += sk
      val grad_new = autoGrad.updateTheta(theta).grad
      var yk = grad_new - grad

      sk = sk.reshape(sk.size, 1) //为了后边的计算
      yk = yk.reshape(yk.size, 1)

      //z1 = (yk' * sk) # a value
      val z1: Double = (yk.t * sk).data(0)
      if (z1 > 0) {
        //z2 = (sk * sk') # a matrix
        val z2 = sk * sk.t

        //z3 = sk * yk' # a matrix
        val z3 = sk * yk.t

        //z4 = yk * sk'
        val z4 = z3.t

        pos = (pos + 1 + m) % m
        if (yk_seq.size < m) {
          yk_seq :+= yk
          sk_seq :+= sk
        } else {
          yk_seq(pos) = yk
          sk_seq(pos) = sk
        }

        val H_k: Double = {
          if (epoch == 1) {
            1.0
          } else {
            (sk_seq(pos).t * yk_seq(pos) / (yk_seq(pos).t * yk_seq(pos))).data(0)
          }
        }

        val cache_size = min(m, yk_seq.size)
        var alpha_seq: Seq[Double] = Nil
        var q = grad_new.reshape(grad_new.size, 1)

        0 until yk_seq.size foreach {
          i => {
            val realpos = (pos - i + cache_size) % cache_size
            val alphai = (1.0 / (yk_seq(realpos).t * sk_seq(realpos)) * (sk_seq(realpos).t * q)).data(0)

            alpha_seq +:= alphai //方便下次遍历，所以逆序添加元素
            q -= alphai * yk_seq(realpos)
          }
        }

        var r: DenseMatrix[Double] = H_k * q

        0 until yk_seq.size foreach {
          i => {
            val realpos = (pos + i + cache_size) % cache_size
            val beta = (1.0 / (yk_seq(realpos).t * sk_seq(realpos)) * yk_seq(realpos).t * r)
            r += sk_seq(realpos) * (alpha_seq(i) - beta)
          }
        }

        dk = -r.reshape(dk.rows, dk.cols)
        variables.setParam("dk", dk)
        variables.setParam("pos", pos)
      }

      variables.setParam("theta", theta)
      variables.setParam("yk_seq", yk_seq)
      variables.setParam("sk_seq", sk_seq)
    }
  }

}
