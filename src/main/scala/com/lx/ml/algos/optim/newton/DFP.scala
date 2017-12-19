package com.lx.ml.algos.optim.newton

import breeze.linalg.{DenseMatrix, Matrix, sum}
import breeze.numerics.abs
import com.lx.ml.algos.loss.{LogLoss, LossFunction}
import com.lx.ml.algos.metrics.ClassificationMetrics
import com.lx.ml.algos.norm.{DefaultNormFunction, L1NormFunction, L2NormFunction}
import com.lx.ml.algos.optim.Optimizer
import com.lx.ml.algos.utils.{AutoGrad, Param}

import scala.reflect.ClassTag
import scala.util.control.Breaks.{break, breakable}

/**
  *
  * @project scalaML
  * @author lx on 6:37 PM 12/12/2017
  */

/**
  * see <https://www.cnblogs.com/qw12/p/5656765.html> and <http://blog.csdn.net/itplus/article/details/21896453>
  */
class DFP extends Optimizer with Param {

  protected def init_param(): DFP = {
    setParams(Seq(
      "lambda" -> 0.15, // 正则化权重,weigjht decay，未设置时无用
      "verbose" -> false, //打印日志
      "printPeriod" -> 100,
      "penalty" -> "l2", //正则化系数，暂只实现l2
      "iterNum" -> 1000, //迭代轮数
      "loss" -> new LogLoss
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


  def loss = getParam[LossFunction]("loss")

  def penalty = getParam[String]("penalty")

  def iterNum = getParam[Int]("iterNum")

  def printPeriod = getParam[Int]("printPeriod")

  def verbose = getParam[Boolean]("verbose")

  def lambda = getParam[Double]("lambda")


  def set_loss(lossFunction: LossFunction) = setParam[LossFunction]("loss", lossFunction)

  def set_penalty(penalty: String) = setParam[String]("penalty", penalty)

  def set_iterNum(iterNum: Int) = setParam[Int]("iterNum", iterNum)

  def set_printPeriod(printPeriod: Int) = setParam[Int]("printPeriod", printPeriod)

  def set_verbose(verbose: Boolean) = setParam[Boolean]("verbose", verbose)

  def set_lambda(lambda: Double) = setParam[Double]("lambda", lambda)


  private val MIN_POS_EPS = -1e-3 //正定矩阵的约束，修正版的DFP使用，主要是发现直接要求正定，效果很差，但确实大部分时候，sk.t*yk的值又非常接近于0，因此做了一个最小阈值

  private val MIN_ALPHA_LOSS_EPS = 0.01

  //hessian矩阵
  private var Hessian: DenseMatrix[Double] = null

  def penaltyNorm = penalty.toLowerCase match {
    case "l1" => L1NormFunction
    case "l2" => L2NormFunction
    case _ => DefaultNormFunction
  }

  // this is python source code: http://dataunion.org/20714.html
  def fit(X: Matrix[Double], Y: Seq[Double]): DFP = {

    assert(X.rows == Y.size && X.rows > 0)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y = format_y(DenseMatrix.create(Y.size, 1, Y.toArray), loss)
    var theta = DenseMatrix.ones[Double](x.cols, 1)

    Hessian = DenseMatrix.eye[Double](x.cols)

    val autoGrad = new AutoGrad(x, y, theta, loss, penaltyNorm, lambda)
    var J = autoGrad.avgLoss

    var Gradient = autoGrad.avgGrad

    var Dk = -Gradient //n * 1

    breakable {
      for (epoch <- 1 to iterNum) {
        if (sum(abs(Dk)) <= MIN_LOSS) {
          println(s"converged at iter $epoch!")
          val acc = ClassificationMetrics.accuracy_score(predict(X), Y)
          log_print(epoch, acc, J)
          break
        }

        // find alpha that min f(thetaK + alpha * Dk)
        // find optimal [a,b] which contain optimal alpha
        // optimal alpha lead to min{f(theta + alpha*DK)}
        var h = Math.random()
        var alpha = 0.0

        var (alpha1, alpha2) = (0.0, h)
        var (theta1, theta2) = (theta + alpha1 * Dk, theta + alpha2 * Dk)

        var f1 = autoGrad.updateTheta(theta1).avgLoss
        var f2 = autoGrad.updateTheta(theta2).avgLoss


        var Loop = 1

        var a, b = 0.0
        breakable {
          while (Loop > 0) {
//            println(s"find [a,b] loop is $Loop in epoch $epoch: ($alpha1, $alpha2), ($f1, $f2)")
            Loop += 1

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
            val theta3 = theta + alpha3 * Dk
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
          }
        }

        breakable {
          // find optiaml alpha in [a,b] using huang jin fen ge fa
          while (Loop > 0) {
            alpha1 = a + 0.382 * (b - a)
            alpha2 = a + 0.618 * (b - a)

            theta1 = theta + alpha1 * Dk
            theta2 = theta + alpha2 * Dk

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

        val theta_old = theta
        val grad_old = - autoGrad.updateTheta(theta_old).avgGrad
        theta = theta + alpha * Dk
        _weight = theta.toDenseVector

        //update the Hessian matrix

        J = autoGrad.updateTheta(theta).avgLoss

        //here to estimate Hessian'inv
        //sk = ThetaNew - ThetaOld = alpha * inv(H) * Gradient
        val sk = theta - theta_old
        //yk = GradNew - GradOld
        //the grad is average value
        val grad = - autoGrad.avgGrad
        val yk = grad - grad_old reshape(x.cols, 1)

        //z1 = (sk' * yk) # a value
        val z1 = (sk.t * yk).data(0)

        //修正正定
        if (z1 > MIN_POS_EPS) {

          //z2 = (sk * sk') # a matrix
          val z2 = sk * sk.t

          //z3 = (yk' * H * yk) # a value
          //z4 = (H * yk * yk' * H) # a matrix
          val z3 = (yk.t * Hessian * yk).data(0)

          val z4 = Hessian * yk * yk.t * Hessian

          val DHessian = z2 / z1 - z4 / z3
          Hessian += DHessian
          Dk = -Hessian * grad.reshape(x.cols, 1)
        }
        if (verbose) {
          if (epoch % printPeriod == 0 || epoch == iterNum) {
            val acc = ClassificationMetrics.accuracy_score(predict(X), Y)
            log_print(epoch, acc, J)
          }
        }

      }

    }

    this
  }



  override def predict(X: Matrix[Double]): Seq[Double] = super.predict_lr(X)

  override def predict_proba(X: Matrix[Double]): Seq[Double] = super.predict_proba_lr(X)
}
