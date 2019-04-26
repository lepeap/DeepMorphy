using System.Collections.Generic;
using System.Linq;
using DeepMorphy.PreProc;
using DeepMorphy.WordDict;

namespace DeepMorphy
{
    public sealed class MorphAnalyzer
    {
        private IPreProcessor[] _preProcessors;
        private NeuralNet.Processor _net;
        private bool _withTrimAndLower;
        public MorphAnalyzer(bool useEnTags=false, int variantsCount=8, bool withTrimAndLower=true)
        {
            var dict = new Dict(useEnTags);
            _net = new NeuralNet.Processor(useEnTags, false, variantsCount);
            _withTrimAndLower = withTrimAndLower;
            _preProcessors = new IPreProcessor[]
            {
                new NarNumbProc(dict), 
                new DictProc(dict), 
                new RegProc(_net.AvailableChars, useEnTags, 50)
            };
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