using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Plastic.Newtonsoft.Json;
using UnityEngine;

public class MLPParameters
{
    public List<float[,]> coeficients;
    public List<float[]> intercepts;
    public float[] mean;
    public float[] standarDesv;
    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {

        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }


    public void CreateIntercept(int i, int row)
    {

        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }

}


public class MLPModel
{

    MLPParameters param_;
    int cont;
    // MLPParameters mlpParameters;
    public MLPModel(MLPParameters param)
    {
        param_ = param;
        //string jsonFilePath = "Assets/Karting/Scripts/ML/thetas.json";
        //string jsonText = System.IO.File.ReadAllText(jsonFilePath);

        //pesos = JsonConvert.DeserializeObject<float[][][]>(jsonText);

    }
    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(Perception p, Transform transform)
    {

        float[] input = new float[8];
        PerceptionInfo[] perceptionInfo = p.Perceptions;
        for (int i = 0; i < perceptionInfo.Length; i++)
        {
            if (perceptionInfo[i].detected)
            {
                input[i] = perceptionInfo[i].hit.distance;
            }
            else input[i] = 10; //Se pone 10 por que es el valor por defecto de los rayos 
        }
        input[5] = transform.position.x;
        input[6] = transform.position.y;
        input[7] = transform.position.z;



        MLPFeedFoward sklearnMLP = new MLPFeedFoward();
        float[] scaled = ScaleData(input);
        float[] output = sklearnMLP.FeedFow(param_, scaled);

        return output;
    }
    public float[] ScaleData(float[] data)
    {

        float[] scaledData = new float[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            if (i > param_.mean.Length - 1 || i > param_.standarDesv.Length - 1)
            {
                scaledData[i] = (data[i] - 0.4f) / 0.2f; //Escala por defecto cuando no hay valores
            }
            else scaledData[i] = (data[i] - param_.mean[i]) / param_.standarDesv[i];

        }

        return scaledData;
    }

    private float CalculateMean(float[] data)
    {
        float sum = 0;

        foreach (float value in data)
        {
            sum += value;
        }

        return sum / data.Length;
    }

    private float CalculateStandardDeviation(float[] data)
    {
        float mean = CalculateMean(data);
        float sumSquaredDiff = 0;

        foreach (float value in data)
        {
            float diff = value - mean;
            sumSquaredDiff += diff * diff;
        }

        float variance = sumSquaredDiff / data.Length;
        return (float)Math.Sqrt(variance);
    }



    /// <summary>
    /// Implements the conversion of the output value to the action label. 
    /// Depending on what actions you have chosen or saved in the dataset, and in what order, the way it is converted will be one or the other.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public Labels ConvertIndexToLabel(int index)
    {

        switch (index)
        {
            case 0:
                Debug.Log("ACCELERATE");
                return Labels.ACCELERATE;
                
            case 1:
                Debug.Log("LEFT_ACCELERATE");
                return Labels.LEFT_ACCELERATE;
            case 2:
                Debug.Log("RIGHT_ACCELERATE");
                return Labels.RIGHT_ACCELERATE;


        }
        return Labels.NONE;
        //TODO: implement the conversion from index to actions.

    }

    public Labels Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        Labels label = ConvertIndexToLabel(index);
        return label;
    }

    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        max = output[0];
        int index = 0;
        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}
/*CLASE QUE REALIZA FEED FOWARD */


public class MLPFeedFoward : MonoBehaviour
{
    public float[] FeedFow(MLPParameters param_, float[] input)
    {

        float[] inputWithBIas = AdbiasetoInput(input);

        for (int i = 0; i < param_.coeficients.Count; i++)
        {
            //Multiplicacion de la matriz de entrada con el sesgo , junto con los coeficientes (con el sesgo de la capa añadido)
            float[] z = MatMUl(inputWithBIas, AddBiastoLayer(param_.coeficients[i], param_.intercepts[i]));
            float[] zRelu = new float[z.Length];
            int length = z.Length;
            //Aplicacion de la funcion Relu
            if (i < param_.coeficients.Count - 1)
            {
                // Aplicación de la función ReLU si no estamos en la última capa
                for (int x = 0; x < length; x++)
                {
                    zRelu[x] = Math.Max(0.0f, z[x]);
                }
            }
            else
            {
            
                float sumExp = 0.0f;

                // Cálculo de la suma de los exponentes para la función softmax
                for (int x = 0; x < z.Length; x++)
                {
                    sumExp += (float)Math.Exp(z[x]);
                }

                // Aplicación de la función softmax
                for (int x = 0; x < z.Length; x++)
                {
                    zRelu[x] = (float)Math.Exp(z[x]) / sumExp;
                }
            }
         

            inputWithBIas = zRelu; //Actualizacion de la entrada con la salida de la capa z tras aplicar relu
            if (i < param_.coeficients.Count - 1) //Si no nos encontramos en la ultima capa añadimos el sesgo
            {
                inputWithBIas = AdbiasetoInput(inputWithBIas);
            }
        }

        return inputWithBIas;
    }


    public float[,] AddBiastoLayer(float[,] mat1, float[] Row)
    {
        int mat1fil = mat1.GetLength(0);
        int columnas = mat1.GetLength(1);

        // MatResult = mat1 but 1 row more
        float[,] matResult = new float[mat1fil + 1, columnas];

        for (int i = 0; i < mat1fil; i++)
        {
            for (int j = 0; j < columnas; j++)
            {
                matResult[i + 1, j] = mat1[i, j];
            }
        }
        // New row at start
        for (int j = 0; j < columnas; j++)
        {
            matResult[0, j] = Row[j];
        }

        return matResult;
    }

    public float[] AdbiasetoInput(float[] input)
    {
        int length = input.Length;
        float[] newArray = new float[length + 1];

        newArray[0] = 1.0f; //Biase

        for (int i = 0; i < length; i++)
        {
            newArray[i + 1] = input[i];
        }

        return newArray;
    }

    public float[] MatMUl(float[] A, float[,] B)
    {
        int rowsA = 1;
        int colsA = A.Length;

        int rowsB = B.GetLength(0);
        int colsB = B.GetLength(1);

        if (colsA != rowsB) throw new ArgumentException("dim1!=dim2");

        float[] result = new float[colsB];

        for (int i = 0; i < colsB; i++)
        {
            float sum = 0.0f;


            for (int j = 0; j < colsA; j++)
            {
                sum += A[j] * B[j, i];

            }

            result[i] = sum;
        }

        return result;
    }

    public float[,] TransposeMatrix(float[,] coeficients, float[] intercepts)
    {
        float[,] original = AddBiastoLayer(coeficients, intercepts);
        int rows = original.GetLength(0);
        int cols = original.GetLength(1);

        float[,] Transpose = new float[cols, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Transpose[j, i] = original[i, j];
            }
        }

        return Transpose;
    }


}

public class MLAgent : MonoBehaviour
{
    public enum ModelType { MLP = 0 }
    public TextAsset text;
    public ModelType model;
    public bool agentEnable;

    private MLPParameters mlpParameters;
    private MLPModel mlpModel;
    private Perception perception;

    // Start is called before the first frame update
    void Start()
    {


        if (agentEnable)
        {


            //Debug.Log(file);
            string file = text.text;
            if (model == ModelType.MLP)
            {
                mlpParameters = LoadParameters(file);
                mlpModel = new MLPModel(mlpParameters);
            }
            //Debug.Log("Parameters loaded " + mlpParameters);
            perception = GetComponent<Perception>();
        }
    }



    public KartGame.KartSystems.InputData AgentInput()
    {
        Labels label = Labels.NONE;
        switch (model)
        {
            case ModelType.MLP:

                float[] outputs = this.mlpModel.FeedForward(perception, this.transform);
                label = this.mlpModel.Predict(outputs);
                break;
        }
        KartGame.KartSystems.InputData input = Record.ConvertLabelToInput(label);
        return input;
    }

    public static string TrimpBrackers(string val)
    {
        val = val.Trim();
        val = val.Substring(1);
        val = val.Substring(0, val.Length - 1);
        return val;
    }

    public static int[] SplitWithColumInt(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        int[] result = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = values[i].Trim();
            if (values[i].StartsWith("'"))
                values[i] = values[i].Substring(1);
            if (values[i].EndsWith("'"))
                values[i] = values[i].Substring(0, values[i].Length - 1);
            result[i] = int.Parse(values[i]);
        }
        return result;
    }

    public static float[] SplitWithColumFloat(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = float.Parse(values[i], System.Globalization.CultureInfo.InvariantCulture);

        }
        return result;
    }

    public static MLPParameters LoadParameters(string file)
    {
        string[] lines = file.Split("\n");
        int num_layers = 0;
        MLPParameters mlpParameters = null;
        int currentParameter = -1;
        int[] currentDimension = null;
        bool coefficient = false;
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            line = line.Trim();
            if (line != "")
            {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "num_layers")
                {
                    num_layers = int.Parse(val);
                    mlpParameters = new MLPParameters(num_layers);
                }

                else
                {
                    if (num_layers <= 0)
                        Debug.LogError("Format error: First line must be num_layers");
                    else
                    {
                        if (name == "parameter")
                            currentParameter = int.Parse(val);
                        else if (name == "dims")
                        {
                            val = TrimpBrackers(val);
                            currentDimension = SplitWithColumInt(val);
                        }
                        else if (name == "name")
                        {
                            if (val.StartsWith("coefficient"))
                            {
                                coefficient = true;
                                int index = currentParameter / 2;
                                Debug.Log(currentParameter);
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else
                            {
                                coefficient = false;
                                mlpParameters.CreateIntercept(currentParameter, currentDimension[0]);
                            }

                        }
                        else if (name == "values")
                        {
                            val = TrimpBrackers(val);
                            float[] parameters = SplitWithColumFloat(val);

                            for (int index = 0; index < parameters.Length; index++)
                            {
                                if (coefficient)
                                {
                                    int row = index / currentDimension[1];
                                    int col = index % currentDimension[1];
                                    mlpParameters.SetCoeficiente(currentParameter, row, col, parameters[index]);
                                }
                                else
                                {
                                    mlpParameters.SetIntercept(currentParameter, index, parameters[index]);
                                }
                            }
                        }
                        else if (name == "mean")
                        {
                            val = TrimpBrackers(val);
                            float[] media = SplitWithColumFloat(val);
                            mlpParameters.mean = media;
                        }
                        else if (name == "stDev")
                        {
                            val = TrimpBrackers(val);
                            float[] desvi = SplitWithColumFloat(val);
                            mlpParameters.standarDesv = desvi;

                        }
                    }
                }
            }
        }
        return mlpParameters;
    }
}
