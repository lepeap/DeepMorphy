using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;

namespace DeepMorphy.NeuralNet
{
    class Processor
    {
        private TfNeuralNet _net;
        private Config _config;
        private int _k;


        public Processor(bool useEnTags = false, bool bigModel = false, int k = 8)
        {
            _config = new Config(useEnTags, bigModel);
            _net = new TfNeuralNet(_config.OpDic, _config.GramOpDic, bigModel);
            _k = k;
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
        public IEnumerable<Token> Parse(IEnumerable<string> words)
        {
            var srcMas = words.ToArray();
            
            _vectorize(srcMas, 
                       out int maxLength, 
                       out List<int[]> indexes,
                       out List<int> values,
                       out int[] seqLens
                       );
            
            
            var result = _net.Classify(maxLength, srcMas.Length, indexes, values, seqLens, _k);
            for (int i = 0; i < srcMas.Length; i++)
            {
                yield return new Token(
                    srcMas[i],
                    Enumerable.Range(0, _k)
                        .Select(j => new TagsCombination(
                            _config.ClsDic[result.ResultIndexes[i, j]],
                            result.ResultProbs[i, j])
                        )
                        .ToArray(),
                    result.GramProbs.ToDictionary(
                        kp => kp.Key,
                        kp => new TagCollection(
                            Enumerable.Range(0, _config[kp.Key].NnClassesCount)
                            .Select(j => new Tag(
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