using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using TensorFlow;

namespace DeepMorphy.NeuralNet
{
    class TfNeuralNet : IDisposable
    {
        public class Result
        {
            public Dictionary<string, float[,]> GramProbs;
            public int[,] ResultIndexes;
            public float[,] ResultProbs;
        }
        private readonly TFGraph _graph;
        private readonly TFSession _session;
        private readonly string[] _gramKeys;
        private readonly string[] _ops;
        private readonly string _xOpName = "X";
        private readonly string _seqLenOpName = "SeqLen";
        private readonly string _kOpName = "Classification/TopK";
        private readonly int _gramsCount;
        private readonly int _mainClsValsIndex;
        private readonly int _mainClsProbsIndex;

        public TfNeuralNet(string mainClsOp, 
                         IDictionary<string, string> gramOpDic,
                         bool bigModel)
        {
            _gramKeys = gramOpDic.Select(x=>x.Key).ToArray();
            _gramsCount = _gramKeys.Length;
            
            var ops = gramOpDic.Values.ToList();
            ops.Add($"{mainClsOp}:0");
            ops.Add($"{mainClsOp}:1");
            
            _mainClsProbsIndex = _gramsCount;
            _mainClsValsIndex = _gramsCount + 1;
            _ops = ops.ToArray();
            _graph = new TFGraph();
            _graph.Import(_getModel(bigModel));
            _session = new TFSession(_graph);
            
            
        }

        private byte[] _getModel(bool bigModel)
        {
            var modelKey = bigModel ? "big" : "small"; 
            var resourceName = $"DeepMorphy.NeuralNet.frozen_model_{modelKey}.pb";
            using (Stream stream = Utils.GetResourceStream(resourceName))
            {
                using (MemoryStream ms = new MemoryStream())
                {
                    stream.CopyTo(ms);
                    return ms.ToArray();
                }
            }
        }

        public Result Classify(int[,] words, int[] seqLens, int k)
        {
            var runner = _session.GetRunner();
            runner.AddInput(_xOpName, words);
            runner.AddInput(_seqLenOpName, seqLens);
            runner.AddInput(_kOpName, k);

            var res = runner.Fetch(_ops).Run();

            var gramProbs = new Dictionary<string, float[,]>();
            for (int i = 0; i < _gramKeys.Length; i++)
                gramProbs[_gramKeys[i]] = (float[,]) res[i].GetValue();
            
            
            return new Result()
            {
                GramProbs = gramProbs,
                ResultProbs = (float[,]) res[_mainClsProbsIndex].GetValue(),
                ResultIndexes = (int[,]) res[_mainClsValsIndex].GetValue()
            };

        }

        public void Dispose()
        {
            _graph?.Dispose();
            _session?.Dispose();
        }
    }
}