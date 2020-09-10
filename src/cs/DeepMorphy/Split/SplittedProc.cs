using System.Collections.Generic;
using System.Linq;
using DeepMorphy.NeuralNet;
using DeepMorphy.WordDict;

namespace DeepMorphy.Split
{
    internal abstract class SplittedProc<I, R> 
    {
        private readonly string[] _processorKeys;
        public SplittedProc(IEnumerable<I> input, MorphAnalyzer morph)
        {
            Input = input.ToArray();
            Result = new R[Input.Length];
            Morph = morph;
            _processorKeys = new string[Input.Length];
            _fillProcessors();
            Processors = morph.Processors;
            Net = morph.Net;
            CorrectionDict = morph.CorrectionDict;
        }
        
        protected I[] Input { get; }
        
        protected R[] Result { get; }
        
        protected MorphAnalyzer Morph { get; }

        protected  IMorphProcessor[] Processors { get; } 
        
        protected NetworkProc Net { get; }
        
        protected Dict CorrectionDict { get; }
        

        protected abstract int _GetTagIndex(I input);

        protected IEnumerable<string> GetProcessorKeys()
        {
            return _processorKeys.Distinct();
        }

        protected IEnumerable<(I input, int index)> GetForProcessor(string processor)
        {
            for (int i = 0; i < Input.Length; i++)
            {
                if (_processorKeys[i] == processor)
                {
                    yield return (Input[i], i);
                }
            }
        }
        
        private void _fillProcessors()
        {
            for (int i = 0; i < Input.Length; i++)
            {
                var tagIndex = _GetTagIndex(Input[i]);
                _processorKeys[i] = TagHelper.TagProcDic[tagIndex];
            }
        }
    }
}