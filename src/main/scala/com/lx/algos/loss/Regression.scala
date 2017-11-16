package com.lx.algos.loss

//所有的回归损失函数都能供分类使用，但是分类的不能直接拿来为回归使用
trait Regression extends LossFunction

class SquaredLossFunction extends Regression {
  //Squared loss traditional used in linear regression.
  override def loss(p: Double, y: Double): Double = {
    val d = p - y
    d * d * 0.5
  }

  override def dLoss(p: Double, y: Double): Double = p - y
}

class Huber(c: Double = 0.1) extends Regression {
  /** Huber regression loss
    * *
    * Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    * linear in for large errors).
    * *
    * https://en.wikipedia.org/wiki/Huber_Loss_Function
    * 对异常值不敏感，本质上是加了约束的平方损失函数
    * c表示对在一定范围误差内的按照平方误差，超过这个范围，按照线性更新
    */
  override def loss(p: Double, y: Double): Double = {
    val r = Math.abs(p - y)

    if (r <= c) 0.5 * r * r
    else c * (r - 0.5 * c)
  }

  override def dLoss(p: Double, y: Double): Double = {
    val r = p - y
    val abs_r = Math.abs(r)

    if (abs_r <= c) 0
    else if (r > 0) c
    else -c
  }
}

class EpsilonInsensitive(epsilon: Double = 0.1) extends Regression {
  /** Epsilon-Insensitive loss (used by SVR).
    * *
    * loss = max(0, |y - p| - epsilon)
    * *
    * epsilon表示误差在一定范围内忽略
    */
  override def loss(p: Double, y: Double): Double = Math.max(0, Math.abs(p - y) - epsilon)

  override def dLoss(p: Double, y: Double): Double = {
    if (y - p > epsilon) -1
    else if(p - y > epsilon) 1
    else 0
  }
}

class SquaredEpsilonInsensitive(epsilon: Double = 0.1) extends Regression {
  /** Epsilon-Insensitive loss.
    * *
    * loss = max(0, |y - p| - epsilon)^2
    * */
  override def loss(p: Double, y: Double): Double = {
    val z = Math.max(0, Math.abs(y - p) - epsilon)
    z * z
  }

  override def dLoss(p: Double, y: Double): Double = {
    val z = y - p
    if (z > epsilon) -2 * (z - epsilon)
    else if (z < -epsilon) 2 * (-z - epsilon)
    else 0
  }
}
