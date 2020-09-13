using System;
using System.Collections.Generic;
using System.Linq;
using DeepMorphy.Model;
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
        internal readonly bool _withTrimAndLower;
        
        /// <summary>
        /// Создает морфологический анализатор. В идеале лучше использовать его как синглтон,
        /// при создании объекта какое-то время уходит на загрузку словарей и сети.
        /// --------------------
        /// Initializes morphology analyzer
        /// </summary>
        /// <param name="withLemmatization">
        /// Вычислять ли леммы слов при разборе слов (по умолчанию - false). Если нужна лемматизация при разборе, то необходимо выставить в true,
        /// иначе лучше не включать (без флага работает быстрее).
        /// --------------------
        /// Perform lemmatization on for each tag in while parsing
        /// </param>
        /// <param name="useEnGrams">
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
            bool useEnGrams = false,
            bool withTrimAndLower = true,
            int maxBatchSize = 4096)
        {
            if (maxBatchSize <= 0)
            {
                throw new ArgumentException("Batch size must be greater than 0.");
            }
            
            _withTrimAndLower = withTrimAndLower;
            UseEnGrams = useEnGrams;
            TagHelper = new TagHelper(useEnGrams);
            Net = new NeuralNet.NetworkProc(TagHelper, maxBatchSize, withLemmatization, useEnGrams);
            CorrectionDict = new Dict("dict_correction");
            Processors = new IMorphProcessor[]
            {
                new RegProc(Net.AvailableChars, 50),
                new NumberProc(),
                new NarNumberProc(),
                new DictProc("dict")
            };
        }
        
        /// <summary>
        /// Используются ли английские названия граммем
        /// </summary>
        public bool UseEnGrams { get; }
        
        /// <summary>
        /// Объект для работы с тегами
        /// </summary>
        public TagHelper TagHelper { get; }

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

                var gramDic = taglist.Any(t => TagHelper.TagProcDic[t.Id] != "nn")
                    ? _mergeGramDic(taglist, netTpl.gramDic)
                    : netTpl.gramDic;

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
                yield return new MorphInfo(netTpl.srcWord, resTags, gramDic, UseEnGrams);
            }
        }

        public IEnumerable<MorphInfo> Parse(params string[] words)
        {
            return Parse((IEnumerable<string>) words);
        }
        
        /// <summary>
        /// Производит лемматизация одного слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <param name="tag">Тег слова</param>
        /// <returns>Лемма</returns>
        public string Lemmatize(string word, Tag tag)
        {
            var req = new[]
            {
                new LemTask(word, tag)
            };
            return Lemmatize(req).First();
        }
        
        /// <summary>
        /// Производит лемматизацию слов
        /// </summary>
        /// <param name="tasks">Слова для лемматизации</param>
        /// <returns>Перечисление получившихся лемм</returns>
        public IEnumerable<string> Lemmatize(IEnumerable<LemTask> tasks)
        {
            if (_withTrimAndLower)
            {
                tasks = tasks.Select(t => new LemTask(t.word.Trim().ToLower(), t.tag));
            }
            return new LemmaProc(tasks, this).Process();
        }
        
        /// <summary>
        /// Производит постановку слов в другую форму
        /// </summary>
        /// <param name="tasks">Слова для изменения формы</param>
        /// <returns>Слова в запрашиваемых формах (если не удалось поставить слово в форму, то null)</returns>
        public IEnumerable<string> Inflect(IEnumerable<InflectTask> tasks)
        {
            if (_withTrimAndLower)
            {
                tasks = tasks.Select(t => new InflectTask(t.word.Trim().ToLower(), t.wordTag, t.resultTag));
            }
            return new InflectProc(tasks, this).Process();
        }

        /// <summary>
        /// Возвращает все формы данного слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <param name="tag">Тег слова</param>
        /// <returns>Перечисление, тег - слово</returns>
        public IEnumerable<(Tag tag, string text)> Lexeme(string word, Tag tag)
        {
            if (_withTrimAndLower)
            {
                word = word.Trim().ToLower();
            }
            
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
        
        private Dictionary<string, GramCategory> _mergeGramDic(List<Tag> tags, Dictionary<string, GramCategory> netGrams)
        {
            var rezCats = new Dictionary<string, GramCategory>();
            foreach (var gram in GramInfo.GramsInfo)
            {
                var gramName = UseEnGrams ? gram.KeyEn : gram.KeyRu;
                var tagGrams = tags.Select(x => x[gramName])
                                .Where(x => x != null)
                                .Distinct()
                                .ToArray();

                if (tagGrams.Length == 0)
                {
                    tagGrams = gram.Classes
                        .Select(x => UseEnGrams ? x.KeyEn : x.KeyRu)
                        .ToArray();
                }
                        
                var power = 1.0f / (tagGrams.Length + 1);
                var curRezDic = tagGrams.ToDictionary(x => x, x => new Gram(x, power));
                foreach (var netGram in netGrams[gramName].Grams)
                {
                    if (!curRezDic.ContainsKey(netGram.Key))
                    {
                        var resGram = new Gram(netGram.Key, netGram.Power * power);
                        curRezDic[netGram.Key] = resGram;
                    }
                }

                var gramRez = curRezDic.Select(x => x.Value).OrderByDescending(x => x.Power).ToArray();
                rezCats[gramName] = new GramCategory(gramRez);
            }
            return rezCats;
        }
    }
}