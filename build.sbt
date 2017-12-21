
enablePlugins(PackPlugin)

name := "scalaML"

version := "0.1"

scalaVersion := "2.11.7"

crossScalaVersions := Seq("2.11.2", "2.10.3")

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

javacOptions ++= Seq("-Xlint:unchecked")

libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",

  // Native libraries are not included by default. add this if you want them (asvr of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",

//   https://mvnrepository.com/artifact/joda-time/joda-time
  "joda-time" % "joda-time" % "2.9.9",

  "edu.cmu.ml.rtw" %% "matt-util" % "2.3.2"
)


resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

