import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer}
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import scala.util.control.Breaks._

import org.apache.log4j.Logger
import org.apache.log4j.Level


/**
  * Created by cai on 04.07.17.
  */

object emTrain {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val smoothing = 0.007D
  def main(args: Array[String]){
    val spark = SparkSession.builder().appName("naiveBayes").getOrCreate()
    import spark.implicits._
    val train = spark.read.parquet("/home/cai/DM/output-train")   // label: [0.0, 19.0]
    val test = spark.read.parquet("/home/cai/DM/output-test")
    //pre-processing
    val indexer = new StringIndexer().setInputCol("topic").setOutputCol("label")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("keywords")
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")//.setNumFeatures(500)
    val win = Window.partitionBy("label").orderBy("label")

    val naiveBayes = new NaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol("label").setFeaturesCol("features")
    val preproPipe = new Pipeline()
      .setStages(Array(stopWordsRemover, hashingTF, indexer))
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val model = preproPipe.fit(train)
    val corpusTest = model.transform(test)
    val corpusTrain = model.transform(train)//.repartition(200)
    //    val unlabeledTrain = corpusTrain.drop("label").cache()
    val numCate = corpusTest.groupBy()
      .max(indexer.getOutputCol).first().getDouble(0).toInt

    val NumSupervisedItems = Array[Int](
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    for(numSupervisedItems <- NumSupervisedItems){
      val labeledTrain = corpusTrain
        .withColumn("rowNum", row_number().over(win))
        .where($"rowNum" <= numSupervisedItems).drop("rowNum")
        .repartition(10)
      val unlabeledTrain = corpusTrain
        .withColumn("rowNum", row_number().over(win))
        .where($"rowNum" > numSupervisedItems)
        .drop("rowNum").drop("label")
        .repartition(10)
      //prediction using EM & NaiveBayes
      val model = emTrainNaiveBayes(labeledTrain, unlabeledTrain, numCate, spark)
      val prediction = model.transform(corpusTest)
      val acc = evaluator.evaluate(prediction)
      //prediction using supervised classification
      val supervisedModel = naiveBayes.fit(labeledTrain)
      val supervisedPrediction = supervisedModel.transform(corpusTest)
      val accSup = evaluator.evaluate(supervisedPrediction)
      //show accuracy
      println(s"number of supervised items: $numSupervisedItems")
      println(s"EM accuracy: $acc")
      println(s"Supervised accuracy: $accSup")
      println()
    }

  }

  def emTrainNaiveBayes(labeledTrain: DataFrame, unlabeledTrain: DataFrame, numCate: Int, sparkSession: SparkSession) : NaiveBayesModel = {
    val naiveBayes = new NaiveBayes()
      .setSmoothing(smoothing)
      .setModelType("multinomial")
      .setLabelCol("label")
      .setFeaturesCol("features")
    val maxEpoch = 20
    val minImprovement = 0.000001D
    var lastModel = naiveBayes.fit(labeledTrain)        //features, label ==> prediction
    //val numCat: Int = labeledTrain.groupBy("label").count().count().toInt;
    var lastLogProb = 1.0D / 0.0

    breakable {
      for (epoch <- 1 to maxEpoch) {
        val prediction = lastModel.transform(unlabeledTrain)
          .withColumnRenamed(naiveBayes.getPredictionCol, "label")
          .drop(naiveBayes.getRawPredictionCol)
          .drop(naiveBayes.getProbabilityCol)
        //using the labeled data
        val combinedTrainingSet = prediction.union(labeledTrain)//.repartition(10)
        val model = naiveBayes.fit(combinedTrainingSet)
        //training finished

        //calculate log probability
        var modelLogProb = 0.0D
        var dataLogProb = 0.0D
        modelLogProb = modelLogCal(model)
        //        println(s"epoch = $epoch     modelLogProb = $modelLogProb")
        dataLogProb = dataLogCal(combinedTrainingSet.select("features", "label"), model, sparkSession)
        val logProb = modelLogProb + dataLogProb
        //calculate the improvement
        val relativeDiff = relativeDif(logProb, lastLogProb)
        println(s"epoch = $epoch     modelLogProb = $modelLogProb     dataLogProb = $dataLogProb" +
          s"     logProb: $logProb     improvement: $relativeDiff")
        lastModel = model                               //feature, label1, prediction
        if (relativeDiff < minImprovement) {
          println("Maximum reached.")
          break
        }
        lastLogProb = logProb
      }
    }
    lastModel
    //return lastModel
  }

  def dataLogCal(data: DataFrame, model: NaiveBayesModel, sparkSession: SparkSession): Double ={
    val accumulator = sparkSession.sparkContext.doubleAccumulator("dataLog Accumulator")
    data.foreach(row => {
      val sparseVector = row.getAs[Vector](0)
      val category = row.getDouble(1).toInt
      sparseVector.foreachActive((ind, _) => {
        accumulator.add(model.theta(category,ind))   //calculate P(wss|cat)
      })
      accumulator.add(model.pi(category))   // log(P(wss|cat)*P(cat))
    })  // sum of all log(P(wss|cat)*P(cat))
    accumulator.value
  }

  def modelLogCal(model: NaiveBayesModel): Double = {
    var modelLogProb = 0.0D
    for(i <- 0 until model.theta.numRows){
      modelLogProb += model.pi(i) * (-1.0D)
    }
    for(i <- 0 until model.theta.numCols){    //number of tokens
      for(j <- 0 until model.theta.numRows){  //number of categories
        modelLogProb += model.theta(j,i) * (-1.0D)
      }
    }
    modelLogProb
    //return modelLogProb
  }

  def relativeDif(x: Double, y: Double): Double = {
    val absDif = math.abs(x - y)
    val absSum = math.abs(x) + math.abs(y)
    (2.0*absDif) / absSum
  }

  def log2(e:Double): Double ={
    //return math.log(e)/math.log(2)
    math.log(e)/math.log(2)
  }
}
