import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}


import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
object MLibTrails {
  def main(args :Array[String]){
   
    val conf = new SparkConf().setAppName("MLIB Trails").setMaster("local")
   
    val ss = SparkSession.builder. master("local").appName("Mlib Trails").getOrCreate()
    val sentenceData = ss.createDataFrame(Seq(
  (0.0, "Hi I heard about Spark"),
  (0.0, "I wish Java could use case classes"),
  (1.0, "Logistic regression models are neat")
)).toDF("label", "sentence")


val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val wordsData = tokenizer.transform(sentenceData)
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
val featurizedData = hashingTF.transform(wordsData)


//featurizedData.collect.foreach(println)
//val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//val idfModel = idf.fit(featurizedData)
//val rescaledData = idfModel.transform(featurizedData)
//rescaledData.select("label", "features").collect.foreach(println)


//println("*******word2vec***********")
//
//val word2Vec = new Word2Vec().setInputCol("words").setOutputCol("result").setVectorSize(3).setMinCount(0)
//val model = word2Vec.fit(featurizedData)
//model.getVectors.collect.foreach(println)
//model.transform(featurizedData).collect.foreach(println)



//println("********** N - Gram **********")
//
//val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
//ngram.transform(wordsData).collect.foreach(println)

println("***********OneHotEncoder***********")
val df = ss.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "d"),
  (4, "e"),
  (5, "f")
)).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(df)
val indexed = indexer.transform(df)
indexed.show()
val encoder = new OneHotEncoder().setInputCol("categoryIndex").setOutputCol("categoryVec")
val encoded = encoder.transform(indexed)
encoded.show()

  }
}