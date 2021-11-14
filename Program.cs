using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MongoDB.Bson;
using MongoDB.Driver.Core;
using MongoDB.Driver;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;


namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_esp.txt");
        string line;
        #region Acciones de la clase principal, se lee el ID del usuario
        static void Main()
        {
            MLContext mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            Console.WriteLine("Escriba el ID del usuario");
            string usuario = Console.ReadLine();
            UseModelWithSingleItem(mlContext, model,usuario);
           // UseModelWithBatchItems(mlContext, model);
        }
        #endregion
        #region Entrenando y dividiendo el conjunto de datos (80-20)
        public static TrainTestData LoadData(MLContext mlContext) 
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
        #endregion
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName:"Features"));
            Console.WriteLine("==========Creando y entrenando el modelo=========");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("==================Fin del entrenamiento============================");
            Console.WriteLine();
            return model;
        }
        #region Evalúa el modelo entrenado con el 20% de datos o conjunto de prueba
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet) 
        {
            Console.WriteLine("=============Evaluando precisión del modelo y F1 Score==================================");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Evaluación de métricas");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Precisión (Acc): {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve: P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine(" ==============Fin de la evaluación del modelo ====================");
        }
        #endregion

        #region Toma un documento de la base de datos MongoDB, lee el comentario, lo pondera, escribe la ponderación y valoración dada por el modelo
        public static void UseModelWithSingleItem(MLContext mlContext, ITransformer model, string usuario)
         {
            try
            {
                var settings = MongoClientSettings.FromConnectionString("mongodb+srv://telemine39:gustavoramos@cluster0.yy1mi.mongodb.net/myFirstDatabase?retryWrites=true&w=majority");
                var client = new MongoClient(settings);
                var database = client.GetDatabase("EvenbriteHackaton");
                var collection = database.GetCollection<BsonDocument>("MueblesAlfonsoMarina");
                var filter = Builders<BsonDocument>.Filter.Eq("id", usuario);
                var BsonDoc = collection.Find(filter).FirstOrDefault();
                string userName = BsonDoc["usuario"].AsString;
                string inputComment = BsonDoc["comentario"].AsString;
                var act = BsonDoc["actitud"].AsDouble;
                var rap = BsonDoc["rapidez"].AsDouble;
                var crea = BsonDoc["creatividad"].AsDouble;
                var ser = BsonDoc["servicio"].AsDouble;
                double ponderacion;
                PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
                SentimentData sampleStatement = new SentimentData
                {
                    SentimentText = inputComment
                };
                var resultPrediction = predictionFunction.Predict(sampleStatement);
                Console.WriteLine();
                Console.WriteLine("================Prediciendo el comentario del usuario =====================");
                Console.WriteLine();
                string predictedValue = (Convert.ToBoolean(resultPrediction.Prediction) ? "Positivo" : "Negativo");
                string Prob = Convert.ToString(resultPrediction.Probability);
                ponderacion = ((act + rap + crea + ser) / 4)*Convert.ToDouble(Prob);
                String pondStr = ponderacion.ToString("0.00");
                UpdateData(usuario, "ponderacion", pondStr);
                UpdateData(usuario, "valoracion", predictedValue);
                Console.WriteLine("Usuario: " + userName);
                Console.WriteLine($"Comentario: {resultPrediction.SentimentText} | Predicción (P/N): {predictedValue} | Positividad: {Prob}");
                Console.WriteLine("==============Fin de las predicciones ==========");
             }
            catch
            {
                Console.WriteLine("Error, el ID de usuario es incorrecto o no existe");
            }
       }
        #endregion
        #region Actualiza el documento, escribe en los campos valoracion: positivo/negativo y en ponderacion el promedio global
        static void UpdateData(string id, string field,string valueUp) 
        {
            var settings = MongoClientSettings.FromConnectionString("mongodb+srv://telemine39:gustavoramos@cluster0.yy1mi.mongodb.net/myFirstDatabase?retryWrites=true&w=majority");
            var client = new MongoClient(settings);
            var database = client.GetDatabase("EvenbriteHackaton");
            var collection = database.GetCollection<BsonDocument>("MueblesAlfonsoMarina");
            var myFilter = Builders<BsonDocument>.Filter.Eq("id", id);
            var update = Builders<BsonDocument>.Update.Set(field, valueUp);
            collection.UpdateOne(myFilter,update);
        }
        #endregion


        /*public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model) 
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                { 
                    SentimentText = "Pésimo servicio"
                },
                new SentimentData
                { 
                    SentimentText = "Servicio muy deficiente"
                }
            };
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = model.Transform(batchComments);
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();
            Console.WriteLine("==================Predicción usando entradas aleatorias ==================================");
            Console.WriteLine();
            foreach (SentimentPrediction prediction in predictedResults) 
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability}");
            }
            Console.WriteLine("====== Fin de las predicciones ======");
        } */
    }
}
