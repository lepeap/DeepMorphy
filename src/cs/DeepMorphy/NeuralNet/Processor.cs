using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DeepMorphy.NeuralNet
{
    internal class Processor
    {
        private bool _withLemmatization;
        private readonly TfNeuralNet _net;
        private readonly Config _config;
        private readonly int _maxBatchSize;
        private const int K = 4;

        public Processor(int maxBatchSize, bool withLemmatization = false, bool useEnGrams = false, bool bigModel = false)
        {
            _maxBatchSize = maxBatchSize;
            _withLemmatization = withLemmatization;
            _config = new Config(useEnGrams, bigModel);
            _net = new TfNeuralNet(_config.OpDic, _config.GramOpDic, bigModel, withLemmatization);
        }

        public char[] AvailableChars => _config.CharToId.Keys.ToArray();
        
        public IEnumerable<MorphInfo> Parse(IEnumerable<string> words)
        {
            foreach (var batch in _batchify(words, _maxBatchSize)){
                var srcMas = batch.ToArray();

                _vectorizeWords(srcMas,
                    out int maxLength,
                    out List<int[]> indexes,
                    out List<int> values,
                    out int[] seqLens
                );

                var result = _net.Classify(maxLength, srcMas.Length, indexes, values, seqLens);
                for (int i = 0; i < srcMas.Length; i++)
                {
                    yield return new MorphInfo(
                        srcMas[i],
                        Enumerable.Range(0, K)
                            .Select(j => new Tag(
                                    _config.ClsDic[result.ResultIndexes[i, j]],
                                    result.ResultProbs[i, j],
                                    lemma: _withLemmatization 
                                        ? _getLemma(srcMas[i], result.Lemmas, i,j, result.ResultIndexes[i, j]) 
                                        : null,
                                    classIndex: result.ResultIndexes[i, j]
                                )
                            )
                            .ToArray(),
                        result.GramProbs.ToDictionary(
                            kp => kp.Key,
                            kp => new GramCategory(
                                Enumerable.Range(0, _config[kp.Key].NnClassesCount)
                                    .Select(j => new Gram(
                                        _config[kp.Key, j],
                                        result.GramProbs[kp.Key][i, j++]
                                    ))
                                    .OrderByDescending(x => x.Power)
                                    .ToArray()
                            )
                        )
                    );
                }      
            }
        }

        public IEnumerable<string> Lemmatize(IEnumerable<(string word, Tag tag)> srcItems)
        {
            foreach (var batch in _batchify(srcItems, _maxBatchSize))
            {
                var words = batch.Select(tpl => tpl.word).ToArray();
                var classes = batch.Select(tpl => tpl.tag.ClassIndex.Value).ToArray();
                _vectorizeWords(words,
                    out int maxLength,
                    out List<int[]> indexes,
                    out List<int> values,
                    out int[] seqLens
                );
                var result = _net.Lemmatize(maxLength, words.Length, indexes, values, seqLens, classes);
            }

            return null;
        }

        public void _vectorizeWords(string[] srcMas,
            out int maxLength,
            out List<int[]> indexes,
            out List<int> values,
            out int[] seqLens
        )
        {
            maxLength = 0;
            indexes = new List<int[]>();
            values = new List<int>();
            seqLens = new int[srcMas.Length];
            for (int i = 0; i < srcMas.Length; i++)
            {
                for (int j = 0; j < srcMas[i].Length; j++)
                {
                    
                    indexes.Add(new int[]{i,j});
                    var curChar = srcMas[i][j];
                    int rezId;
                    if (_config.CharToId.ContainsKey(curChar))
                        rezId = _config.CharToId[curChar];
                    else
                        rezId = _config.UndefinedCharId;

                    values.Add(rezId);
                }

                if (maxLength < srcMas[i].Length)
                    maxLength = srcMas[i].Length;

                seqLens[i] = srcMas[i].Length;
            }
        }
        
        private IEnumerable<IEnumerable<T>> _batchify<T>(IEnumerable<T> srcItems, int batchSize)
        {
            IEnumerable<T> items = srcItems.ToArray();
            while (items.Any())
            {
                yield return items.Take(batchSize);
                items = items.Skip(batchSize);    
            }
        }

        private string _getLemma(string sourceWord, int[,,] nnRes, int wordIndex, int kIndex, int mainCls)
        {
            if (_config.LemmaSameWordClasses.Contains(mainCls))
                return sourceWord;
                
            int cIndex = 0;
            var maxLength = nnRes.GetLength(2);
            var sb = new StringBuilder();
            while (cIndex < maxLength)
            {
                var cVal = nnRes[wordIndex, kIndex, cIndex];
                if (cVal == _config.EndCharIndex)
                    break;
                
                if (cVal == _config.UndefinedCharId)
                    return null;

                sb.Append(_config.IdToChar[cVal]);
                cIndex++;
            }
            return sb.ToString();
        }
    }
}