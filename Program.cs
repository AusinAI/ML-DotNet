using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MyMLApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create MLContext
            MLContext mlContext = new MLContext();

            // Sample data for training
            var data = mlContext.Data.LoadFromEnumerable(new[]
            {
                new ModelInput { Feature1 = 0.2f, Feature2 = 0.8f, Label = true },
                new ModelInput { Feature1 = 0.5f, Feature2 = 0.3f, Label = false },
                new ModelInput { Feature1 = 0.9f, Feature2 = 0.1f, Label = true }
            });

            // Define data preprocessing and model pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(ModelInput.Feature1), nameof(ModelInput.Feature2))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: nameof(ModelInput.Label)));

            // Train the model
            var model = pipeline.Fit(data);

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            // Make predictions
            var inputData = new ModelInput { Feature1 = 0.6f, Feature2 = 0.4f }; // Example input data
            var prediction = predictionEngine.Predict(inputData);

            Console.WriteLine($"Prediction: {prediction.Prediction}"); // Predicted class (0 or 1)
            Console.WriteLine($"Probability: {prediction.Probability}"); // Probability of belonging to class 1

            // You can use the prediction in your application logic
        }
    }

    // Define input data class
    public class ModelInput
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public bool Label { get; set; } // Label column
    }

    // Define output data class
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
