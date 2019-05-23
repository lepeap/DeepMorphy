using System;
using System.Collections.Generic;
using System.Linq;
using DeepMorphy.PreProc;
using DeepMorphy.WordDict;

namespace DeepMorphy
{
    /// <summary>
    /// Main class of morphology analyzer
    /// </summary>
    public sealed class MorphAnalyzer
    {
        private readonly bool _withTrimAndLower;
        private readonly IPreProcessor[] _preProcessors;
        private readonly NeuralNet.Processor _net;
        
        /// <summary>
        /// Initializes morphology analyzer
        /// </summary>
        /// <param name="withLemmatization">Perform lemmatization for each tag</param>
        /// <param name="useEnGrams">if true returns english gramme names otherwise russian</param>
        /// <param name="withTrimAndLower">if true analyzer trims and makes words lowercase before processing</param>
        /// <param name="withPreprocessors">use additional preprocessors before nn</param>
        /// <param name="maxBatchSize">max batch size for neural network</param>
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
            var dict = new Dict(useEnGrams, withLemmatization);
            _net = new NeuralNet.Processor(maxBatchSize, withLemmatization, useEnGrams, false);
            _withTrimAndLower = withTrimAndLower;
            if (withPreprocessors)
                _preProcessors = new IPreProcessor[]
                {
                    new NarNumbProc(dict, withLemmatization),
                    new DictProc(dict),
                    new RegProc(_net.AvailableChars, useEnGrams, 50, withLemmatization)
                };
            else
                _preProcessors = new IPreProcessor[0];
        }
        
        /// <summary>
        /// Calculates morphology information for words
        /// </summary>
        /// <param name="words">Words to process</param>
        /// <returns>Morphology information for each word</returns>
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
    }
}