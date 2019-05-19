using System;
using System.Collections.Generic;
using System.Linq;
using DeepMorphy.PreProc;
using DeepMorphy.WordDict;

namespace DeepMorphy
{
    public sealed class MorphAnalyzer
    {
        private readonly bool _withTrimAndLower;
        private IPreProcessor[] _preProcessors;
        private NeuralNet.Processor _net;
        public MorphAnalyzer(bool withLemmatization = false, 
                             bool useEnTags=false, 
                             bool withTrimAndLower=true,
                             bool withPreprocessors=true,
                             int maxBatchSize=4096)
        {
            if (maxBatchSize <= 0)
            {
                throw new ArgumentException("Batch size must be greater than 0.");
            }
            var dict = new Dict(useEnTags);
            _net = new NeuralNet.Processor(maxBatchSize, withLemmatization, useEnTags, false);
            _withTrimAndLower = withTrimAndLower;
            if (withPreprocessors)
                _preProcessors = new IPreProcessor[]
                {
                    new NarNumbProc(dict, withLemmatization),
                    new DictProc(dict),
                    new RegProc(_net.AvailableChars, useEnTags, 50, withLemmatization)
                };
            else
                _preProcessors = new IPreProcessor[0];
        }

        public IEnumerable<Token> Parse(IEnumerable<string> words)
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