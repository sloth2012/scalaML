package com.lx.algos.ml.optim.newton

import breeze.linalg.{DenseMatrix, Matrix, min, sum}
import breeze.numerics.abs
import com.lx.algos.ml.loss.{LogLoss, LossFunction}
import com.lx.algos.ml.metrics.ClassificationMetrics
import com.lx.algos.ml.norm.{DefaultNormFunction, L1NormFunction, L2NormFunction}
import com.lx.algos.ml.optim.Optimizer
import com.lx.algos.ml.utils.{AutoGrad, MatrixTools, Param, TimeUtil}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 4:53 PM 28/12/2017
  */


//实现见Numerical Optimization第7章179页前后
//该方法采用不精确的一维搜索，即满足Wolfe-Powell准则，见Numerical Optimization79页
class LBFGS extends Optimizer with Param {


  protected def init_param(): LBFGS = {
    setParams(Seq(
      "lambda" -> 0.15, // 正则化权重,weigjht decay，未设置时无用
      "verbose" -> false, //打印日志
      "printPeriod" -> 100,
      "penalty" -> "l2", //正则化系数，暂只实现l2
      "iterNum" -> 1000, //迭代轮数
      "loss" -> new LogLoss,
      "m" -> 8 //保存最近的m次信息，sk和yk，一般在3到20之间
    ))

    this
  }


  override def setParam[T: ClassTag](name: String, value: T) = {
    super.setParam[T](name, value)
    this
  }

  override def setParams[T <: Iterable[(String, Any)]](params: T) = {
    super.setParams[T](params)
    this
  }

  init_param()

  //参照<http://blog.csdn.net/mytestmy/article/details/16903537>
  lazy val c1 = 0.1 //通常在(0,0.5)之间,指的是ρ
  lazy val c2 = 0.4 //0.1相当于线性搜索，0.9相当于弱的线性搜索，通常取0.4，其应该在(rho,1)之间，指σ

  def loss = getParam[LossFunction]("loss")

  def penalty = getParam[String]("penalty")

  def iterNum = getParam[Int]("iterNum")

  def printPeriod = getParam[Int]("printPeriod")

  def verbose = getParam[Boolean]("verbose")

  def lambda = getParam[Double]("lambda")

  def m = getParam[Int]("m", 8)

  def set_m(m: Int) = setParam[Int]("m", m)

  def set_loss(lossFunction: LossFunction) = setParam[LossFunction]("loss", lossFunction)

  def set_penalty(penalty: String) = setParam[String]("penalty", penalty)

  def set_iterNum(iterNum: Int) = setParam[Int]("iterNum", iterNum)

  def set_printPeriod(printPeriod: Int) = setParam[Int]("printPeriod", printPeriod)

  def set_verbose(verbose: Boolean) = setParam[Boolean]("verbose", verbose)

  def set_lambda(lambda: Double) = setParam[Double]("lambda", lambda)

  def penaltyNorm = penalty.toLowerCase match {
    case "l1" => L1NormFunction
    case "l2" => L2NormFunction
    case _ => DefaultNormFunction
  }

  private val MIN_POS_EPS = 0.0 //正定矩阵的约束，修正版的BFGS使用，主要是发现直接要求正定，效果很差，但确实大部分时候，yk.t*sk的值又非常接近于0，因此做了一个最小阈值

  private val MIN_ALPHA_LOSS_EPS = 0.01

  var yk_seq: ArrayBuffer[DenseMatrix[Double]] = ArrayBuffer.empty[DenseMatrix[Double]]

  var sk_seq: ArrayBuffer[DenseMatrix[Double]] = ArrayBuffer.empty[DenseMatrix[Double]]

  var pos: Int = -1 //表示yk和sk的起始点，即第一个元素的位置，用以每次只抛弃修改一个位置

  override def fit(X: Matrix[Double], y: Seq[Double]): Optimizer = {

    assert(X.rows == y.size && X.rows > 0)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y_format = format_y(DenseMatrix.create(y.size, 1, y.toArray), loss)

    breakable {
      var converged = false
      var Dk: DenseMatrix[Double] = null
      //I, 单位矩阵
      val I = DenseMatrix.eye[Double](x.cols)
      var J: Double = 0
      for (epoch <- 1 to iterNum) {
        val (new_x, new_y) = MatrixTools.shuffle(x, y_format)
        val autoGrad = new AutoGrad(new_x, new_y, _theta, loss, penaltyNorm, lambda)

        val theta_old = _theta
        val grad_old = autoGrad.avgGrad

        if (epoch == 1) { //init
          Dk = -grad_old //n * 1
          J = autoGrad.avgLoss
        }

        if (sum(abs(Dk)) <= MIN_LOSS || converged) {
          println(s"converged at iter ${epoch - 1}!")
          val acc = ClassificationMetrics.accuracy_score(predict(X), y)
          log_print(epoch - 1, acc, J)
          break
        }

        // find alpha that min f(thetaK + alpha * Dk)
        // find optimal [a,b] which contain optimal alpha
        // optimal alpha lead to min{f(theta + alpha*DK)}
        var h = Math.random() //步长
        var alpha = 0.0

        var (alpha1, alpha2) = (0.0, h)
        var (theta1, theta2) = (_theta + alpha1 * Dk, _theta + alpha2 * Dk)

        var f1 = autoGrad.updateTheta(theta1).avgLoss
        var f2 = autoGrad.updateTheta(theta2).avgLoss


        var Loop = 1

        var a, b = 0.0
        breakable {
          while (Loop > 0) {
            //            println(s"find [a,b] loop is $Loop in epoch $epoch: ($alpha1, $alpha2), ($f1, $f2)")

            if (f1 > f2) h *= 2
            else {
              h *= -1

              val tmp_alpha = alpha1
              alpha1 = alpha2
              alpha2 = tmp_alpha

              val tmp_f = f1
              f1 = f2
              f2 = tmp_f
            }

            val alpha3 = alpha2 + h
            val theta3 = _theta + alpha3 * Dk
            val f3 = autoGrad.updateTheta(theta3).avgLoss

            //            println(s"f3 - f2 is ${f3 - f2}")

            if (f3 > f2) {
              a = Math.min(alpha1, alpha3)
              b = Math.max(alpha1, alpha3)
              break
            } else {
              alpha1 = alpha2
              alpha2 = alpha3

              f1 = f2
              f2 = f3
            }

            Loop += 1
          }
        }

        breakable {
          // find optiaml alpha in [a,b] using huang jin fen ge fa
          while (Loop > 0) {
            alpha1 = a + 0.382 * (b - a)
            alpha2 = a + 0.618 * (b - a)

            theta1 = _theta + alpha1 * Dk
            theta2 = _theta + alpha2 * Dk

            f1 = autoGrad.updateTheta(theta1).avgLoss
            f2 = autoGrad.updateTheta(theta2).avgLoss

            if (f1 > f2) a = alpha1
            if (f1 < f2) b = alpha2

            if (abs(a - b) <= MIN_ALPHA_LOSS_EPS) {
              alpha = (a + b) / 2
              break
            }
          }
        }

        //        println(s"optimal alpha in epoch $epoch is $alpha")

        _theta = _theta + alpha * Dk

        val new_J = autoGrad.updateTheta(_theta).avgLoss

        //here to estimate Hessian'inv
        //sk = ThetaNew - ThetaOld = alpha * inv(H) * Gradient
        val sk = _theta - theta_old // shape=n*1
        //yk = GradNew - GradOld
        //the grad is average value
        val grad = autoGrad.avgGrad
        val yk = grad - grad_old // shape=n*1

        //z1 = (yk' * sk) # a value
        val z1 = (yk.t * sk).data(0)


        //修正正定
        if (z1 > MIN_POS_EPS) {

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

          val H_k = {
            if (epoch == 1) {
              1.0
            } else {
              (sk_seq(pos).t * yk_seq(pos) / (yk_seq(pos).t * yk_seq(pos))).data(0)
            }
          } * I

          val cache_size = min(m, yk_seq.size)
          var alpha: Seq[Double] = Nil
          var q = grad

          0 until yk_seq.size foreach {
            i => {
              val realpos = (pos - i + cache_size) % cache_size
              val alphai = (1.0 / (yk_seq(realpos).t * sk_seq(realpos)) * (sk_seq(realpos).t * q)).data(0)

              alpha +:= alphai //方便下次遍历，所以逆序添加元素
              q -= alphai * yk_seq(realpos)
            }
          }

          var r = H_k * q

          0 until yk_seq.size foreach {
            i => {
              val realpos = (pos + i + cache_size) % cache_size
              val beta = (1.0 / (yk_seq(realpos).t * sk_seq(realpos)) * yk_seq(realpos).t * r).data(0)
              r += sk_seq(realpos) * (alpha(i) - beta)
            }
          }

          Dk = -r
        }
        if (verbose) {
          if (epoch % printPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, J)
          }
        }

        converged = Math.abs(new_J - J) < MIN_LOSS

        J = new_J
      }
    }

    this
  }

  //Wolfe-Powell准则实现版本,部分参数未提供修改接口,c1/c2
  def fit_Wolfe_Powell(X: Matrix[Double], y: Seq[Double]): Optimizer = {

    assert(X.rows == y.size && X.rows > 0)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y_format = format_y(DenseMatrix.create(y.size, 1, y.toArray), loss)


    breakable {
      var converged = false
      var Dk: DenseMatrix[Double] = null
      //I, 单位矩阵
      val I = DenseMatrix.eye[Double](x.cols)
      var J: Double = 0 //记录当前的损失

      val newThetaByAlphaFunc = (alpha: Double) => _theta + alpha * Dk //新的weight函数
      val zoomFunc = (alpha_lo: Double, alpha_hi: Double, grad_zero: Double, f_zero: Double, autoGrad: AutoGrad) => {
        var loop = 1
        var low = alpha_lo
        var high = alpha_hi
        var alpha = low

        breakable {
          while (loop > 0) {


            //防止区间过小
            if (abs(high - low) < MIN_ALPHA_LOSS_EPS) {
              break
            }
            //初始化，需要用到插值方法，这里用了二点二次插值
            alpha = low - 0.5 * (low - high) / (1 - (autoGrad.updateTheta(newThetaByAlphaFunc(low)).avgLoss - autoGrad.updateTheta(newThetaByAlphaFunc(high)).avgLoss) /
              ((autoGrad.updateTheta(newThetaByAlphaFunc(low)).avgGrad.t * Dk).data(0) * (low - high)))
            //二分插值
            //            alpha = (low + high) / 2

//                        println(s"zoom loop $loop alpha is $alpha in ($low, $high)")
            val f = autoGrad.updateTheta(newThetaByAlphaFunc(alpha)).avgLoss
            if (f > f_zero + c1 * alpha * grad_zero || f >= autoGrad.updateTheta(newThetaByAlphaFunc(low)).avgLoss) {
              high = alpha
            }
            else {
              val grad = (autoGrad.avgGrad.t * Dk).data(0)
              if (abs(grad) <= c2 * abs(grad_zero)) {
                break
              }
              if (grad * (high - low) >= 0) {
                high = low
              }
              low = alpha
            }

            loop += 1
          }
        }
        alpha
      }

      for (epoch <- 1 to iterNum) {

        val (new_x, new_y) = MatrixTools.shuffle(x, y_format)
        val autoGrad = new AutoGrad(new_x, new_y, _theta, loss, penaltyNorm, lambda)

        val theta_old = _theta
        val grad_old = autoGrad.avgGrad

        if (epoch == 1) { //init
          Dk = -grad_old //n * 1
          J = autoGrad.avgLoss
        }

        if (sum(abs(Dk)) <= MIN_LOSS || converged) {
          println(s"converged at iter ${epoch - 1}!")
          val acc = ClassificationMetrics.accuracy_score(predict(X), y)
          log_print(epoch - 1, acc, J)
          break
        }

        // find alpha that min f(thetaK + alpha * Dk)
        // find optimal [a,b] which contain optimal alpha
        // optimal alpha lead to min{f(theta + alpha*DK)}

        val rnd = new Random(TimeUtil.currentMillis)
        val alpha1: Double = 1e-3
        val alpha2: Double = 1000

        var alpha = rnd.nextDouble() * alpha2

        var loop = 1

        var f_last = J
        var alpha_last = alpha1

        val grad_zero = (grad_old.t * Dk).data(0)


        breakable {
          while (loop > 0) {

//            println(s"loop $loop alpha is $alpha")

            val f_new = autoGrad.updateTheta(newThetaByAlphaFunc(alpha)).avgLoss

            //J means f0
            if (f_new > J + (c1 * alpha * grad_zero) || (loop > 1 && f_new >= f_last)) {
              alpha = zoomFunc(alpha_last, alpha, grad_zero, J, autoGrad)
              break
            }

            val grad_new = (autoGrad.avgGrad.t * Dk).data(0)
            if (grad_new <= c2 * abs(grad_zero)) {
              break
            }
            if (grad_new >= 0) {
              alpha = zoomFunc(alpha, alpha_last, grad_zero, J, autoGrad)
              break
            }

            f_last = f_new
            alpha_last = alpha

            alpha = alpha + (alpha2 - alpha) * rnd.nextDouble()

            loop += 1
          }
        }

//                println(s"optimal alpha in epoch $epoch is $alpha")

        _theta = _theta + alpha * Dk

        val new_J = autoGrad.updateTheta(_theta).avgLoss

        //here to estimate Hessian'inv
        //sk = ThetaNew - ThetaOld = alpha * inv(H) * Gradient
        val sk = _theta - theta_old // shape=n*1
        //yk = GradNew - GradOld
        //the grad is average value
        val grad = autoGrad.avgGrad
        val yk = grad - grad_old // shape=n*1

        //z1 = (yk' * sk) # a value
        val z1 = (yk.t * sk).data(0)


        //修正正定
        if (z1 > MIN_POS_EPS) {

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

          val H_k = {
            if (epoch == 1) {
              1.0
            } else {
              (sk_seq(pos).t * yk_seq(pos) / (yk_seq(pos).t * yk_seq(pos))).data(0)
            }
          } * I

          val cache_size = min(m, yk_seq.size)
          var alpha: Seq[Double] = Nil
          var q = grad

          0 until yk_seq.size foreach {
            i => {
              val realpos = (pos - i + cache_size) % cache_size
              val alphai = (1.0 / (yk_seq(realpos).t * sk_seq(realpos)) * (sk_seq(realpos).t * q)).data(0)

              alpha +:= alphai //方便下次遍历，所以逆序添加元素
              q -= alphai * yk_seq(realpos)
            }
          }

          var r = H_k * q

          0 until yk_seq.size foreach {
            i => {
              val realpos = (pos + i + cache_size) % cache_size
              val beta = (1.0 / (yk_seq(realpos).t * sk_seq(realpos)) * yk_seq(realpos).t * r).data(0)
              r += sk_seq(realpos) * (alpha(i) - beta)
            }
          }

          Dk = -r
        }
        if (verbose) {
          if (epoch % printPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), y)
            log_print(epoch, acc, J)
          }
        }

        converged = Math.abs(new_J - J) < MIN_LOSS

        J = new_J
      }
    }

    this
  }


  override def predict(X: Matrix[Double]): Seq[Double] = super.predict_lr(X)

  override def predict_proba(X: Matrix[Double]): Seq[Double] = super.predict_proba_lr(X)
}
