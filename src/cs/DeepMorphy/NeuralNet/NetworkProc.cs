using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DeepMorphy.NeuralNet
{
    internal class NetworkProc
    {
        private const int K = 4;
        private readonly bool _withLemmatization;
        private readonly TfNeuralNet _net;
        private readonly int _maxBatchSize;
        private readonly TagHelper _tagHelper;
        private readonly Config _config;

        public NetworkProc(TagHelper tagHelper,
            int maxBatchSize,
            bool withLemmatization = false,
            bool useEnGrams = false)
        {
            _tagHelper = tagHelper;
            _maxBatchSize = maxBatchSize;
            _withLemmatization = withLemmatization;
            _config = new Config(useEnGrams);
            _net = new TfNeuralNet(_config.OpDic, _config.GramOpDic, withLemmatization);
        }

        public char[] AvailableChars => _config.CharToId.Keys.ToArray();

        public IEnumerable<(string srcWord, IEnumerable<Tag> tags, Dictionary<string, GramCategory> gramDic)> Parse(IEnumerable<string> words)
        {
            foreach (var batch in _batchify(words, _maxBatchSize))
            {
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
                    var srcword = srcMas[i];
                    var tags = Enumerable.Range(0, K)
                                         .Select(j => new Tag(
                                                 _tagHelper[result.ResultIndexes[i, j]],
                                                 result.ResultProbs[i, j],
                                                 lemma: _withLemmatization
                                                     ? _getLemma(srcMas[i], result.Lemmas, i, j, result.ResultIndexes[i, j])
                                                     : null,
                                                 id: result.ResultIndexes[i, j]
                                             )
                                         );
                    var gramDic = result.GramProbs.ToDictionary(
                        kp => kp.Key,
                        kp => new GramCategory(
                            Enumerable.Range(0, _config[kp.Key].NnClassesCount)
                                .Select(j => new Gram(
                                    _config[kp.Key, j],
                                    result.GramProbs[kp.Key][i, j++]
                                ))
                                .OrderByDescending(x => x.Power)
                                .ToArray()
                        ));
                    yield return (srcWord: srcword, tags: tags, gramDic: gramDic);
                }
            }
        }

        public IEnumerable<((string word, int tagId) task, string resWord)> Lemmatize(IEnumerable<(string word, int tagId)> srcItems)
        {
            foreach (var batchSrc in _batchify(srcItems, _maxBatchSize))
            {
                var batch = batchSrc.ToArray();
                var words = batch.Select(tpl => tpl.word).ToArray();
                var classes = batch.Select(tpl => tpl.tagId).ToArray();
                _vectorizeWords(words,
                    out int maxLength,
                    out List<int[]> indexes,
                    out List<int> values,
                    out int[] seqLens
                );

                var netRes = _net.Lemmatize(maxLength, words.Length, indexes, values, seqLens, classes);
                for (int i = 0; i < words.Length; i++)
                {
                    yield return (
                        task: batch[i],
                        resWord: _getLemma(words[i], netRes, i, 0, classes[i])
                    );
                }
            }
        }

        public IEnumerable<(int srcTagId, string srcWord, int resTagId, string resWord)> Inflect(IEnumerable<(string word, int tag, int resTag)> srcItems)
        {
            foreach (var batchSrc in _batchify(srcItems, _maxBatchSize))
            {
                var batch = batchSrc.ToArray();
                var words = batch.Select(tpl => tpl.word).ToArray();
                var xClasses = batch.Select(tpl => tpl.tag).ToArray();
                var yClasses = batch.Select(tpl => tpl.resTag).ToArray();
                _vectorizeWords(words,
                    out int maxLength,
                    out List<int[]> indexes,
                    out List<int> values,
                    out int[] seqLens
                );

                var netRes = _net.Inflect(maxLength, words.Length, indexes, values, seqLens, xClasses, yClasses);
                for (int i = 0; i < words.Length; i++)
                {
                    yield return (xClasses[i], words[i], yClasses[i], _decodeWord(netRes, i));
                }
            }
        }

        public IEnumerable<(int tagId, string word)> Lexeme(string word, int tagId)
        {
            var items = _config.InflectTemplatesDic[tagId].Select(rTag => (word, tagId, rTag));
            return  Inflect(items).Select(x => (x.resTagId, x.resWord)).Append((tagId, word));
        }

        private void _vectorizeWords(string[] srcMas,
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
                    indexes.Add(new int[] {i, j});
                    var curChar = srcMas[i][j];
                    int rezId;
                    if (_config.CharToId.ContainsKey(curChar))
                    {
                        rezId = _config.CharToId[curChar];
                    }
                    else
                    {
                        rezId = _config.UndefinedCharId;
                    }

                    values.Add(rezId);
                }

                if (maxLength < srcMas[i].Length)
                {
                    maxLength = srcMas[i].Length;
                }

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
            if (TagHelper.IsLemma(mainCls))
            {
                return sourceWord;
            }

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

        private string _decodeWord(int[,] nnRes, int wordIndex)
        {
            int cIndex = 0;
            var maxLength = nnRes.GetLength(1);
            var sb = new StringBuilder();
            while (cIndex < maxLength)
            {
                var cVal = nnRes[wordIndex, cIndex];
                if (cVal == _config.EndCharIndex)
                {
                    break;
                }

                if (cVal == _config.UndefinedCharId)
                {
                    return null;
                }

                sb.Append(_config.IdToChar[cVal]);
                cIndex++;
            }

            return sb.ToString();
        }
    }
}