using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Schema;
using DeepMorphy.Numb;
using DeepMorphy.Split;
using DeepMorphy.WordDict;

namespace DeepMorphy
{
    /// <summary>
    /// Главный класс морфологического анализатора
    /// --------------------
    /// Main class of morphology analyzer
    /// </summary>
    public sealed class MorphAnalyzer
    {
        private readonly bool _withTrimAndLower;

        /// <summary>
        /// Создает морфологический анализатор. В идеале лучше использовать его как синглтон,
        /// при создании объекта какое-то время уходит на загрузку словарей и сети.
        /// --------------------
        /// Initializes morphology analyzer
        /// </summary>
        /// <param name="withLemmatization">
        /// Вычислять ли леммы слов при разборе слов (по умолчанию - false). Если нужна лемматизация, то необходимо выставить в true,
        /// иначе лучше не включать (без флага работает быстрее).
        /// --------------------
        /// Perform lemmatization on for each tag in while parsing
        /// </param>
        /// <param name="useEnGramNames">
        /// Использовать английские названия граммем и грамматических категорий
        /// --------------------
        /// If true returns english gramme names otherwise russian</param>
        /// <param name="withTrimAndLower">
        /// Производить ли обрезку пробелов и приведение слов к нижнему регистру
        /// --------------------
        /// If true analyzer trims and makes words lowercase before processing
        /// </param>
        /// <param name="maxBatchSize">
        /// Максимальный батч, который скармливается нейронной сети
        /// --------------------
        /// Max batch size for neural network
        /// </param>
        /// <exception cref="ArgumentException">if maxBatchSize is not grater then 0</exception>
        public MorphAnalyzer(bool withLemmatization = false,
            bool useEnGramNames = false,
            bool withTrimAndLower = true,
            int maxBatchSize = 4096)
        {
            if (maxBatchSize <= 0)
            {
                throw new ArgumentException("Batch size must be greater than 0.");
            }
            
            _withTrimAndLower = withTrimAndLower;
            UseEnGramNameNames = useEnGramNames;
            GramHelper = new GramHelper();
            TagHelper = new TagHelper(useEnGramNames, GramHelper);
            Net = new NeuralNet.NetworkProc(TagHelper, maxBatchSize, withLemmatization, useEnGramNames);
            CorrectionDict = new Dict("dict_correction");
            Processors = new IMorphProcessor[]
            {
                new RegProc(Net.AvailableChars, 50),
                new NumberProc(),
                new NarNumberProc(),
                new DictProc("dict")
            };
        }

        public bool UseEnGramNameNames { get; }

        public TagHelper TagHelper { get; }

        public GramHelper GramHelper { get; }

        internal NeuralNet.NetworkProc Net { get; }
        internal IMorphProcessor[] Processors { get; set; }
        internal Dict CorrectionDict { get; set; }

        /// <summary>
        /// Производит морфологический разбор слов
        /// --------------------
        /// Calculates morphology information for words
        /// </summary>
        /// <param name="words">
        /// Слова для анализа
        /// --------------------
        /// Words to process
        /// </param>
        /// <returns>
        /// Результат анализа для каждого слова
        /// --------------------
        /// Morphology information for each word
        /// </returns>
        public IEnumerable<MorphInfo> Parse(IEnumerable<string> words)
        {
            if (_withTrimAndLower)
            {
                words = words.Select(x => x.Trim().ToLower());
            }

            foreach (var netTpl in Net.Parse(words))
            {
                bool ignoreNetworkResult = false;
                var taglist = new List<Tag>();
                for (int i = 0; i < Processors.Length; i++)
                {
                    var curProc = Processors[i];
                    var procResults = curProc.Parse(netTpl.srcWord);
                    if (procResults == null)
                    {
                        continue;
                    }

                    foreach (var procRes in procResults)
                    {
                        var tDic = TagHelper.TagsDic[procRes.tagId];
                        taglist.Add(new Tag(tDic, -1, procRes.tagId, procRes.lemma));
                    }

                    if (curProc.IgnoreNetworkResult)
                    {
                        ignoreNetworkResult = true;
                    }

                    break;
                }

                if (!ignoreNetworkResult)
                {
                    taglist.AddRange(netTpl.tags);
                }

                foreach (var corWord in CorrectionDict.Parse(netTpl.srcWord))
                {
                    if (corWord.ReplaceOther)
                    {
                        var tagItem = taglist.FirstOrDefault(x => x.Id == corWord.TagId);
                        taglist.Remove(tagItem);
                    }

                    var corTag = TagHelper.CreateTagFromId(corWord.TagId, -1, corWord.Lemma);
                    taglist.Add(corTag);
                }

                var resTags = _mergeTagsPower(taglist);
                yield return new MorphInfo(netTpl.srcWord, resTags, netTpl.gramDic, UseEnGramNameNames);
            }
        }

        public IEnumerable<MorphInfo> Parse(params string[] words)
        {
            return Parse((IEnumerable<string>) words);
        }

        public string Lemmatize(string word, Tag tag)
        {
            var req = new[]
            {
                (word: word, tag: tag)
            };
            return Lemmatize(req).First();
        }

        public IEnumerable<string> Lemmatize(IEnumerable<(string word, Tag tag)> words)
        {
            return new LemmaProc(words, this).Process();
        }

        public IEnumerable<string> Inflect(IEnumerable<(string word, Tag wordTag)> words, Tag resultTag)
        {
            var request = words.Select(x => (x.word, x.wordTag, resultTag));
            return Inflect(request);
        }

        public IEnumerable<string> Inflect(IEnumerable<(string word, Tag wordTag, Tag resultTag)> words)
        {
            return new InflectProc(words, this).Process();
        }

        /// <summary>
        /// Возвращает все формы данного слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <param name="tag">Тег слова</param>
        /// <returns>Словарь, тег - слово</returns>
        public IEnumerable<(Tag tag, string text)> Lexeme(string word, Tag tag)
        {
            var procKey = TagHelper.TagProcDic[tag.Id];
            if (procKey == "nn")
            {
                var netRes = Net.Lexeme(word, tag.Id);
                var lexeme = CorrectionDict.Lexeme(word, tag.Id);

                netRes = lexeme != null
                    ? netRes.Where(nw => !lexeme.Any(dw => dw.TagId == nw.tagId && dw.ReplaceOther))
                        .Concat(lexeme.Select(dw => (dw.TagId, dw.Text)))
                    : netRes;

                return netRes.OrderByDescending(x => TagHelper.TagOrderDic[x.tagId])
                    .Select(x => (TagHelper.CreateTagFromId(x.tagId), x.word));
            }

            foreach (var processor in Processors)
            {
                if (processor.Key != procKey)
                {
                    continue;
                }

                var result = processor.Lexeme(word, tag.Id);
                if (result != null)
                {
                    return result.Select(x => (TagHelper.CreateTagFromId(x.tagId), x.text));
                }
            }

            return null;
        }


        private IEnumerable<Tag> _mergeGramProbs(List<Tag> tags)
        {
            var preProcCount = tags.Count(x => (x.Power + 1) < 0.0001);
            if (preProcCount == 0)
            {
                return tags;
            }

            var preProcPower = 1f / (preProcCount + 1);
            foreach (var tag in tags)
            {
                if ((tag.Power + 1) < 0.0001)
                {
                    tag.Power = preProcPower;
                }
                else
                {
                    tag.Power = preProcPower * tag.Power;
                }
            }

            var result = tags.OrderByDescending(x => x.Power);
            return result;
        }

        private IEnumerable<Tag> _mergeTagsPower(List<Tag> tags)
        {
            var preProcCount = tags.Count(x => (x.Power + 1) < 0.0001);
            if (preProcCount == 0)
            {
                return tags;
            }

            var preProcPower = 1f / (preProcCount + 1);
            foreach (var tag in tags)
            {
                if ((tag.Power + 1) < 0.0001)
                {
                    tag.Power = preProcPower;
                }
                else
                {
                    tag.Power = preProcPower * tag.Power;
                }
            }

            var result = tags.OrderByDescending(x => x.Power);
            return result;
        }
    }
}