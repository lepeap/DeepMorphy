using System.Collections.Generic;
using System.Linq;
using DeepMorphy.Model;

namespace DeepMorphy.Split
{
    internal class LemmaProc : SplittedProc<LemTask, string>
    {
        public LemmaProc(IEnumerable<LemTask> input, MorphAnalyzer morph) : base(input, morph)
        {
        }
        
        protected override int _GetTagIndex(LemTask input)
        {
            return input.tag.Id;
        }

        public string[] Process()
        {
            foreach (var procKey in GetProcessorKeys())
            {
                foreach (var proc in Processors.Where(p => p.Key == procKey))
                {
                    foreach (var tpl in GetForProcessor(procKey))
                    {
                        if (Result[tpl.index] != null)
                        {
                            continue;
                        }
                        
                        var res = proc.Lemmatize(tpl.input.word, tpl.input.tag.Id);
                        if (res != null)
                        {
                            Result[tpl.index] = res;
                        }
                    }
                }
            }

            var netTask = GetForProcessor("nn").ToArray();
            if (netTask.Length != 0)
            {
                int i = 0;
                foreach (var netRes in Net.Lemmatize(netTask.Select(tpl => (tpl.input.word, tpl.input.tag.Id))))
                {
                    var lexeme = CorrectionDict.Lexeme(netRes.task.word, netRes.task.tagId);
                    var corResult = lexeme?.FirstOrDefault(w => TagHelper.IsLemma(w.TagId))?.Text;
                    Result[netTask[i++].index] = corResult ?? netRes.resWord;
                }
            }
            
            return Result;
        }
    }
}