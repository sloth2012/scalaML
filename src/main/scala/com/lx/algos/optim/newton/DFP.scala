package com.lx.algos.optim.newton

import breeze.linalg.{DenseMatrix, Matrix, sum}
import breeze.numerics.{abs, pow}
import com.lx.algos.loss.LogLoss
import com.lx.algos.metrics.ClassificationMetrics
import com.lx.algos.optim.Optimizer
import com.lx.algos.utils.Param

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

  val iterNum = 1000
  val verbose = true
  val printPeriod = 2000
  val loss = new LogLoss


  // this is python source code: http://dataunion.org/20714.html
  def fit(X: Matrix[Double], Y: Seq[Double]): DFP = {

    assert(X.rows == Y.size && X.rows > 0)

    weight_init(X.cols)
    val x = input(X).toDenseMatrix
    val y = format_y(DenseMatrix.create(Y.size, 1, Y.toArray), loss)
    var theta = DenseMatrix.ones[Double](x.cols, 1)

    var Hessian = DenseMatrix.eye[Double](x.cols)

    var H: DenseMatrix[Double] = x * theta
    var J = sum(loss.loss(H, y)) / x.rows

    var costJ: Seq[Double] = Nil
    var Gradient = (loss.dLoss(H, y)).t * x / (1.0 * x.rows) reshape(x.cols, 1)

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

        var (f1, f2) = (sum(loss.loss(x * theta1, y)) / x.rows, sum(loss.loss(x * theta2, y)) / x.rows)

        var Loop = 1

        var a, b = 0.0
        breakable {
          while (Loop > 0) {
            println(s"find [a,b]=[$a, $b] loop is $Loop")
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
            val f3 = sum(loss.loss(x * theta3, y)) / x.rows

            println(s"f3 - f2 is ${f3 - f2}")

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
          val e = 0.01
          while (Loop > 0) {
            alpha1 = a + 0.382 * (b - a)
            alpha2 = a + 0.618 * (b - a)

            theta1 = theta + alpha1 * Dk
            theta2 = theta + alpha2 * Dk

            f1 = sum(loss.loss(x * theta1, y)) / x.rows
            f2 = sum(loss.loss(x * theta2, y)) / x.rows

            if (f1 > f2) a = alpha1
            if (f1 < f2) b = alpha2

            if (abs(a - b) <= e) {
              alpha = (a + b) / 2
              break
            }
          }
        }

        println(s"optimal alpha is $alpha")

        val theta_old = theta
        theta = theta + alpha * Dk
        _weight = theta.toDenseVector

        //update the Hessian matrix
        H = x * theta
        J = sum(loss.loss(H, y)) / x.rows
        println(s"Itering $epoch; cost is: $J")
        costJ :+= J

        //here to estimate Hessian'inv
        //sk = ThetaNew - ThetaOld = alpha * inv(H) * Gradient
        val sk = theta - theta_old
        //yk = GradNew - GradOld
        //the grad is average value
        val grad = -loss.dLoss(x * theta, y).t * x / (1.0 * x.rows)
        val grad_old = -loss.dLoss(x * theta_old, y).t * x / (1.0 * x.rows)
        val yk = grad - grad_old reshape(x.cols, 1)


        //z1 = (sk' * yk) # a value
        val z1 = sk.t * yk
        //修正正定
        if (z1(0, 0) > 0) {

          //z2 = (sk * sk') # a matrix
          val z2 = sk * sk.t

          //z4 = (H * yk * yk' * H) # a matrix
          //z3 = (yk' * H * yk) # a value
          val z3 = yk.t * Hessian * yk
          val z4 = Hessian * yk * yk.t * yk

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
