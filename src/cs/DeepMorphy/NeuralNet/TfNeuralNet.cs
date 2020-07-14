using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorFlow;

namespace DeepMorphy.NeuralNet
{
    class TfNeuralNet : IDisposable
    {
        private readonly TFGraph _graph;
        private readonly TFSession _session;
        private readonly string[] _gramKeys;
        
        private readonly string[] _clsOps;
        private readonly string[] _lemOps;
        private readonly string[] _inflectOps;

        private readonly string _xIndexesPlName;
        private readonly string _xValuesPlName;
        private readonly string _xShapePlName;
        private readonly string _seqLenPlName;
        private readonly string _batchSizePlName;
        private readonly string _lemXClassPlName;
        private readonly string _inflectXClassPlName;
        private readonly string _inflectYClassPlName;

        private readonly int _mainClsValsIndex;
        private readonly int _mainClsProbsIndex;
        private readonly int _lemClsIndex;

        private bool _clsWithLemmatization;

        public TfNeuralNet(
            IDictionary<string, string> opDic,
            IDictionary<string, string> gramOpDic,
            bool bigModel,
            bool clsWithLemmatization)
        {
            _xIndexesPlName = opDic["x_ind"];
            _xValuesPlName = opDic["x_val"];
            _xShapePlName = opDic["x_shape"];
            _seqLenPlName = opDic["seq_len"];
            _batchSizePlName = opDic["batch_size"];
            _lemXClassPlName = opDic["lem_cls"];
            _inflectXClassPlName = opDic["inflect_x_cls"];
            _inflectYClassPlName = opDic["inflect_y_cls"];
            _clsWithLemmatization = clsWithLemmatization;

            _gramKeys = gramOpDic.Select(x => x.Key).ToArray();
            var gramsCount = _gramKeys.Length;
            var ops = gramOpDic.Values.ToList();
            var mainClsOp = opDic["res_values"];
            ops.Add($"{mainClsOp}:0");
            ops.Add($"{mainClsOp}:1");
            if (clsWithLemmatization)
                ops.Add(opDic["lem_cls_result"]);
            _clsOps = ops.ToArray();

            _lemOps = new[] {opDic["lem_result"]};
            
            _inflectOps = new[] {opDic["inflect_result"]};

            _mainClsProbsIndex = gramsCount;
            _mainClsValsIndex = _mainClsProbsIndex + 1;
            _lemClsIndex = _mainClsValsIndex + 1;
            _graph = new TFGraph();
            _graph.Import(_getModel(bigModel));
            _session = new TFSession(_graph);
        }

        public ClsResult Classify(
            int maxLength,
            int wordsCount,
            IEnumerable<int[]> indexes,
            IEnumerable<int> values,
            int[] seqLens)
        {
            var runner = _createRunner(maxLength, wordsCount, indexes, values, seqLens);
            var res = runner.Fetch(_clsOps).Run();
            var gramProbs = new Dictionary<string, float[,]>();
            for (int i = 0; i < _gramKeys.Length; i++)
                gramProbs[_gramKeys[i]] = (float[,]) res[i].GetValue();

            return new ClsResult()
            {
                GramProbs = gramProbs,
                ResultProbs = (float[,]) res[_mainClsProbsIndex].GetValue(),
                ResultIndexes = (int[,]) res[_mainClsValsIndex].GetValue(),
                Lemmas = _clsWithLemmatization ? (int[,,]) res[_lemClsIndex].GetValue() : null
            };
        }

        public int[,,] Lemmatize(
            int maxLength,
            int wordsCount,
            IEnumerable<int[]> indexes,
            IEnumerable<int> values,
            int[] seqLens,
            int[] xClasses)
        {
            var runner = _createRunner(maxLength, wordsCount, indexes, values, seqLens);
            runner.AddInput(_lemXClassPlName, xClasses);
            var res = runner.Fetch(_lemOps).Run();
            return (int[,,]) res[0].GetValue();
        }

        public int[,,] Inflect(int maxLength,
            int wordsCount,
            IEnumerable<int[]> indexes,
            IEnumerable<int> values,
            int[] seqLens,
            int[] xClasses,
            int[] yClasses)
        {
            var runner = _createRunner(maxLength, wordsCount, indexes, values, seqLens);
            runner.AddInput(_inflectXClassPlName, xClasses);
            runner.AddInput(_inflectYClassPlName, yClasses);
            var res = runner.Fetch(_inflectOps).Run();
            return (int[,,]) res[0].GetValue();
        }

        public void Dispose()
        {
            _graph?.Dispose();
            _session?.Dispose();
        }

        private TFSession.Runner _createRunner(
            int maxLength,
            int wordsCount,
            IEnumerable<int[]> indexes,
            IEnumerable<int> values,
            int[] seqLens)
        {
            var runner = _session.GetRunner();
            runner.AddInput(_xIndexesPlName, indexes.ToArray());
            runner.AddInput(_xValuesPlName, values.ToArray());
            runner.AddInput(_xShapePlName, new int[] {wordsCount, maxLength});
            runner.AddInput(_seqLenPlName, seqLens);
            runner.AddInput(_batchSizePlName, wordsCount);
            return runner;
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
    }
}