using System;
using System.Collections.Generic;
using System.Linq;
using DeepMorphy.PreProc;
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
        private readonly IPreProcessor[] _preProcessors;
        private readonly NeuralNet.Processor _net;
        
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
                             bool useEnGrams=false, 
                             bool withTrimAndLower=true,
                             bool withPreprocessors=true,
                             int maxBatchSize=4096)
        {
            if (maxBatchSize <= 0)
            {
                throw new ArgumentException("Batch size must be greater than 0.");
            }
            _net = new NeuralNet.Processor(maxBatchSize, withLemmatization, useEnGrams, false);
            _withTrimAndLower = withTrimAndLower;
            if (withPreprocessors)
            {
                var dict = new Dict(useEnGrams, withLemmatization);
                _preProcessors = new IPreProcessor[]
                {
                    new NarNumbProc(dict, withLemmatization),
                    new DictProc(dict),
                    new RegProc(_net.AvailableChars, useEnGrams, 50, withLemmatization)
                };
            }
            else
                _preProcessors = new IPreProcessor[0];
        }
        
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
                for (int i = 0; i < _preProcessors.Length; i++)
                {
                    var preProcResult = _preProcessors[i].Parse(netTok.Text);
                    if (preProcResult != null)
                    {
                        yield return preProcResult;
                        ready = true;
                        break;
                    }
                }
                if (!ready)
                    yield return netTok;
            }
        }

        public IEnumerable<string> Lemmatize(IEnumerable<(string word, Tag wordTag)> words)
        {
            return null;
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
        /// <param name="wordTag">Тег слова</param>
        /// <returns>Словарь, тег - слово</returns>
        public IDictionary<Tag, string> GetAllForms(string word, Tag wordTag)
        {
            return null;
        }
    }
}