using System.Collections.Generic;
using System.Linq;
using DeepMorphy.NeuralNet;
using DeepMorphy.WordDict;

namespace DeepMorphy.Split
{
    internal class LemmaProc : SplittedProc<(string word, Tag tag), string>
    {
        public LemmaProc(IEnumerable<(string word, Tag tag)> input,
                                      IMorphProcessor[] processors,
                                      NetworkProc net,
                                      Dict correctionDict) 
                        : base(input, processors, net, correctionDict)
        {
        }
        
        protected override int _GetTagIndex((string word, Tag tag) input)
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
                    Result[netTask[i].index] = corResult ?? netRes.resWord;
                }
            }
            
            return Result;
        }
    }
}