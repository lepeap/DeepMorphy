using System;
using System.Collections.Generic;
using System.Linq;
using DeepMorphy.Numb;
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
        private readonly IMorphProcessor[] _morphProcessors;
        private readonly NeuralNet.NetworkProc _net;

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
        /// <param name="useEnGrams">
        /// Использовать английские названия граммем и грамматических категорий
        /// --------------------
        /// If true returns english gramme names otherwise russian</param>
        /// <param name="withTrimAndLower">
        /// Производить ли обрезку пробелов и приведение слов к нижнему регистру
        /// --------------------
        /// If true analyzer trims and makes words lowercase before processing
        /// </param>
        /// <param name="withPreprocessors">
        /// Использовать ли препроцессоры перед нейронной сетью (по умолчанию - true).
        /// По идее, всегда должно быть true, false ставится только для тестов
        /// --------------------
        /// Use additional preprocessors before nn
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
            bool withPreprocessors = true,
            int maxBatchSize = 4096)
        {
            if (maxBatchSize <= 0)
            {
                throw new ArgumentException("Batch size must be greater than 0.");
            }
            EnTags = useEnGrams;
            GramHelper = new GramHelper();
            TagHelper = new TagHelper(this);
            _net = new NeuralNet.NetworkProc(TagHelper, maxBatchSize, withLemmatization, useEnGrams);
            _withTrimAndLower = withTrimAndLower;
            
            if (withPreprocessors)
            {
                _morphProcessors = new IMorphProcessor[]
                {
                    new NumberProc(),
                    new DictProc(),
                    new RegProc(_net.AvailableChars, 50)
                };
            }
            else
            {
                _morphProcessors = new IMorphProcessor[0];
            }
        }

        public bool EnTags { get; }
        
        public TagHelper TagHelper { get; }
        
        public GramHelper GramHelper { get; }
        
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
                words = words.Select(x => x.Trim().ToLower());
                
            foreach (var netTok in _net.Parse(words))
            {
                bool ready = false;
                for (int i = 0; i < _morphProcessors.Length; i++)
                {
                    //var preProcResult = _morphProcessors[i].Parse(netTok.Text);
                    //if (preProcResult != null)
                    //{
                    //    yield return preProcResult;
                    //    ready = true;
                    //    break;
                    //}
                }
                //if (!ready)
                //    yield return netTok;
            }
            yield break;
        }

        public IEnumerable<MorphInfo> Parse(params string[] words)
        {
            return Parse((IEnumerable<string>)words);
        }
        
        
        public string Lemmatize(string word, Tag tag)
        {
            var req = new []
            {
                (word: word, tag: tag)
            };
            return Lemmatize(req).First();
        }

        public IEnumerable<string> Lemmatize(IEnumerable<(string word, Tag wordTag)> words)
        {
            return _net.Lemmatize(words);
        }

        public IEnumerable<string> Inflect(IEnumerable<(string word, Tag wordTag)> words, Tag resultTag)
        {
            var request = words.Select(x => (x.word, x.wordTag, resultTag));
            return Inflect(request);
        }
        
        public IEnumerable<string> Inflect(IEnumerable<(string word, Tag wordTag, Tag resultTag)> words)
        {
            yield break;
        }
        
        /// <summary>
        /// Возвращает все формы данного слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <param name="tag">Тег слова</param>
        /// <returns>Словарь, тег - слово</returns>
        public IDictionary<Tag, string> Lexeme(string word, Tag tag)
        {
            return _net.Lexeme(word, tag.TagIndex.Value);
        }
    }
}