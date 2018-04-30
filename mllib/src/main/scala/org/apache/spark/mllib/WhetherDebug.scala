package org.apache.spark.mllib

class WhetherDebug{

}
object WhetherDebug{
  var isDebug = false
  def setDebug(xx: Boolean): Unit ={
    isDebug = xx
  }
}