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


//发现该方法非常依赖theta的初始化，可能会每次训练结果都不一致
//更新了下weight的初始化
class LBFGS(var m: Int = 10, //保存最近的m次信息，sk和yk，一般在3到20之间
            var method: LineSearch = LineSearch.wolfePowell, //一维搜索方法
            var tolerance_grad: Double = 1e-8 //termination tolerance on first order optimality
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
//      println(s"find best alpha is $alpha")

      //##############################################
      //update dk, yk_seq, sk_seq, theta st.
      //##############################################
      var sk_mat = alpha * dk
      theta += sk_mat
      val grad_new = autoGrad.updateTheta(theta).grad
      var yk_mat = grad_new - grad

      //do in each class
//      {
//
//        val cache_size = min(m, yk_seq.size)
//
//        val r_class = {
//          0 until sk_mat.cols map { ci =>
//            if ((yk_mat(::, ci) dot sk_mat(::, ci)) > 0) {
//              val h_k = sk_mat(::, ci) dot yk_mat(::, ci)
//
//              var alpha_seq: Seq[Double] = Nil
//              var rho: Array[Double] = new Array[Double](yk_seq.size)
//              var q = grad_new(::, ci)
//
//              0 until yk_seq.size foreach {
//                i => {
//                  val realpos = (pos - i + cache_size) % cache_size
//                  rho(realpos) = (1.0 / (yk_seq(realpos)(::, ci) dot sk_seq(realpos)(::, ci)))
//                  val alphai: Double = rho(realpos) * (sk_seq(realpos)(::, ci) dot q)
//
//                  alpha_seq +:= alphai //方便下次遍历，所以逆序添加元素
//                  q -= alphai * yk_seq(realpos)(::, ci)
//                }
//
//              }
//
//              var r = h_k * q
//
//              0 until yk_seq.size foreach {
//                i => {
//                  val realpos = (pos + i + cache_size) % cache_size
//                  val beta = rho(realpos) * (yk_seq(realpos)(::, ci) dot r)
//                  r += sk_seq(realpos)(::, ci) * (alpha_seq(i) - beta)
//                }
//              }
//
//              -r
//            }
//            else {
//              dk(::, ci)
//            }
//          }
//        }
//        dk = DenseMatrix.create(dk.rows, dk.cols, r_class.flatMap(_.toArray).toArray)
//        variables.setParam("dk", dk)
//        variables.setParam("pos", pos)
//
//        //更新保存的m个状态
//        pos = (pos + 1 + m) % m
//        if (yk_seq.size < m) {
//          yk_seq :+= yk_mat
//          sk_seq :+= sk_mat
//        } else {
//          yk_seq(pos) = yk_mat
//          sk_seq(pos) = sk_mat
//        }
//      }


      val sk = sk_mat.reshape(sk_mat.size, 1) //为了后边的计算
      val yk = yk_mat.reshape(yk_mat.size, 1)

      //z1 = (yk' * sk) # a value
      val z1: Double = (yk.t * sk).data(0)
      if (z1 > 0) {

        //书中错误，rk应为最新的yk*sk相乘
        //see formula 7.20 P178 in Numerical Optimization
        val H_k: Double = (sk.t * yk / (yk.t * yk)).data(0)

//        println(s"hk is $H_k")

        val cache_size = min(m, yk_seq.size)
        var alpha_seq: Seq[Double] = Nil
        var q = grad_new.reshape(grad_new.size, 1)

        0 until yk_seq.size foreach {
          i => {
            val realpos = (pos - i + cache_size) % cache_size
            val alphai = ((1.0 / (yk_seq(realpos).t * sk_seq(realpos))) * (sk_seq(realpos).t * q)).data(0)

            //              println(s"alphai: $alphai")
            alpha_seq +:= alphai //方便下次遍历，所以逆序添加元素
            q -= alphai * yk_seq(realpos)
          }

        }

        var r: DenseMatrix[Double] = H_k * q

        0 until yk_seq.size foreach {
          i => {
            val realpos = (pos + i + cache_size) % cache_size
            val beta = (1.0 / (yk_seq(realpos).t * sk_seq(realpos))) * yk_seq(realpos).t * r
            r += sk_seq(realpos) * (alpha_seq(i) - beta)
          }
        }
        //更新保存的m个状态
        pos = (pos + 1 + m) % m
        if (yk_seq.size < m) {
          yk_seq :+= yk
          sk_seq :+= sk
        } else {
          yk_seq(pos) = yk
          sk_seq(pos) = sk
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
