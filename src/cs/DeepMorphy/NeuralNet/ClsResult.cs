using System.Collections.Generic;

namespace DeepMorphy.NeuralNet
{
    internal class ClsResult
    {
        public Dictionary<string, float[,]> GramProbs;
        public int[,] ResultIndexes;
        public float[,] ResultProbs;
        public int[,,] Lemmas;
    }
}