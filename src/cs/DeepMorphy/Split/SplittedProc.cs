using System.Collections.Generic;
using System.Linq;
using DeepMorphy.NeuralNet;
using DeepMorphy.WordDict;

namespace DeepMorphy.Split
{
    internal abstract class SplittedProc<I, R> 
    {
        public SplittedProc(IEnumerable<I> input, 
                            IMorphProcessor[] processors, 
                            NetworkProc net,
                            Dict correctionDict)
        {
            Input = input.ToArray();
            Result = new R[Input.Length];
            ProcessorKeys = new string[Input.Length];
            _fillProcessors();
            Processors = processors;
            Net = net;
            CorrectionDict = correctionDict;
        }
        
        public I[] Input { get; }
        
        public R[] Result { get; }
        
        public string[] ProcessorKeys { get; }
        
        public  IMorphProcessor[] Processors { get; } 
        
        public NetworkProc Net { get; }
        
        public Dict CorrectionDict { get; }
        

        protected abstract int _GetTagIndex(I input);

        protected IEnumerable<string> GetProcessorKeys()
        {
            return ProcessorKeys.Distinct();
        }

        protected IEnumerable<(I input, int index)> GetForProcessor(string processor)
        {
            for (int i = 0; i < Input.Length; i++)
            {
                if (ProcessorKeys[i] == processor)
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
                ProcessorKeys[i] = TagHelper.TagProcDic[tagIndex];
            }
        }
    }
}