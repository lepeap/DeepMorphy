using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DeepMorphy.NeuralNet
{
    internal class Processor
    {
        private bool _withLemmatization;
        private readonly TfNeuralNet _net;
        private readonly int _maxBatchSize;
        private const int K = 4;

        public Processor(int maxBatchSize, bool withLemmatization = false, bool useEnGrams = false, bool bigModel = false)
        {
            _maxBatchSize = maxBatchSize;
            _withLemmatization = withLemmatization;
            Config = new Config(useEnGrams, bigModel);
            _net = new TfNeuralNet(Config.OpDic, Config.GramOpDic, bigModel, withLemmatization);
        }
        
        public Config Config { get; }

        public char[] AvailableChars => Config.CharToId.Keys.ToArray();
        
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
                                    Config.ClsDic[result.ResultIndexes[i, j]],
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
                                Enumerable.Range(0, Config[kp.Key].NnClassesCount)
                                    .Select(j => new Gram(
                                        Config[kp.Key, j],
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
                
                var netRes = _net.Lemmatize(maxLength, words.Length, indexes, values, seqLens, classes);
                for (int i = 0; i < words.Length; i++)
                {
                    yield return _getLemma(words[i], netRes, i, 0, classes[i]);
                }
            }
        }

        public IEnumerable<(int tagId, string word)> Inflect(IEnumerable<(string word, int tag, int resTag)> srcItems)
        {
            foreach (var batch in _batchify(srcItems, _maxBatchSize))
            {
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
                    yield return (yClasses[i], _decodeWord(netRes, i));
                }
            }
        }

        public Dictionary<Tag, string> GetAllForms(string word, int tagId)
        {
            var items = this.Config.InflectTemplatesDic[tagId].Select(rTag => (word, tagId, rTag));
            var results = this.Inflect(items);
            var resDic = results.ToDictionary(x => new Tag(Config.ClsDic[x.tagId], (float)1.0, word, tagId), x=>x.word);
            return resDic;
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
                    indexes.Add(new int[]{i,j});
                    var curChar = srcMas[i][j];
                    int rezId;
                    if (Config.CharToId.ContainsKey(curChar))
                        rezId = Config.CharToId[curChar];
                    else
                        rezId = Config.UndefinedCharId;

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

        private string _getLemma(string sourceWord, int[,,] nnRes, int wordIndex, int kIndex, long mainCls)
        {
            if (Config.LemmaSameWordClasses.Contains(mainCls))
                return sourceWord;
                
            int cIndex = 0;
            var maxLength = nnRes.GetLength(2);
            var sb = new StringBuilder();
            while (cIndex < maxLength)
            {
                var cVal = nnRes[wordIndex, kIndex, cIndex];
                if (cVal == Config.EndCharIndex)
                    break;
                
                if (cVal == Config.UndefinedCharId)
                    return null;

                sb.Append(Config.IdToChar[cVal]);
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
                if (cVal == Config.EndCharIndex)
                {
                    break;
                }

                if (cVal == Config.UndefinedCharId)
                {
                    return null;
                }

                sb.Append(Config.IdToChar[cVal]);
                cIndex++;
            }
            return sb.ToString();
        }
    }
}