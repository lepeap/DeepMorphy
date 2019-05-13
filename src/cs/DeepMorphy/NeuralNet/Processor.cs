using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;
using System.Text;

namespace DeepMorphy.NeuralNet
{
    class Processor
    {
        private readonly TfNeuralNet _net;
        private readonly Config _config;
        private const int K = 4;

        public Processor(bool withLemmatization = false, bool useEnTags = false, bool bigModel = false)
        {
            _config = new Config(useEnTags, bigModel);
            _net = new TfNeuralNet(_config.OpDic, _config.GramOpDic, bigModel, withLemmatization);
        }

        public char[] AvailableChars => _config.CharToId.Keys.ToArray();

        public void _vectorize(string[] srcMas,
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

        private string _parseLemma(int[,,] src, int wordIndex, int kIndex)
        {
            int cIndex = 0;
            var maxLength = src.GetLength(2);
            var sb = new StringBuilder();
            while (cIndex < maxLength)
            {
                var cVal = src[wordIndex, kIndex, cIndex];

                if (cVal == _config.EndCharIndex)
                    break;
                
                if (cVal == _config.UndefinedCharId)
                    return null;

                sb.Append(_config.IdToChar[cVal]);
                cIndex++;
            }
            return sb.ToString();
        }
        
        public IEnumerable<Token> Parse(IEnumerable<string> words, bool lemmatize=true)
        {
            var srcMas = words.ToArray();
            
            _vectorize(srcMas, 
                       out int maxLength, 
                       out List<int[]> indexes,
                       out List<int> values,
                       out int[] seqLens
                       );
            
            
            var clsResult = _net.Classify(maxLength, srcMas.Length, indexes, values, seqLens, K);
            var result = _processTokenResult(srcMas, clsResult);
            return result;
        }


        private IEnumerable<Token> _processTokenResult(string[] srcMas, TfNeuralNet.Result result)
        {
            for (int i = 0; i < srcMas.Length; i++)
            {
                yield return new Token(
                    srcMas[i],
                    Enumerable.Range(0, K)
                        .Select(j => new Tag(
                                _config.ClsDic[result.ResultIndexes[i, j]],
                                result.ResultProbs[i, j],
                                lemma: _parseLemma(result.Lemmas, i,j),
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
}