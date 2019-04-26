using System;
using System.Collections.Generic;
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
            _net = new TfNeuralNet(_config.MainClsOp, _config.GramOpDic, bigModel);
            _k = k;
        }

        public char[] AvailableChars => _config.CharToId.Keys.ToArray();
        public IEnumerable<Token> Parse(IEnumerable<string> words)
        {
            var srcMas = words.ToArray();
            var maxLen = srcMas.Max(x => x.Length);
            var wordsVectors = new int[srcMas.Length, maxLen];
            var seqLens = new int[srcMas.Length];
            for (int i = 0; i < srcMas.Length; i++)
            {
                for (int j = 0; j < srcMas[i].Length; j++)
                {
                    var curChar = srcMas[i][j];
                    int rezId;
                    if (_config.CharToId.ContainsKey(curChar))
                        rezId = _config.CharToId[curChar];
                    else
                        rezId = _config.UndefinedCharId;

                    wordsVectors[i, j] = rezId;
                }

                seqLens[i] = srcMas[i].Length;
            }

            var result = _net.Classify(wordsVectors, seqLens, _k);
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