using System.Collections.Generic;
using System.Linq;
using DeepMorphy.NeuralNet;
using DeepMorphy.WordDict;

namespace DeepMorphy.Split
{
    internal class InflectProc : SplittedProc<(string word, Tag wordTag, Tag resultTag), string>
    {
        public InflectProc(IEnumerable<(string word, Tag wordTag, Tag resultTag)> input,
            IMorphProcessor[] processors,
            NetworkProc net,
            Dict correctionDict)
            : base(input, processors, net, correctionDict)
        {
        }

        protected override int _GetTagIndex((string word, Tag wordTag, Tag resultTag) input)
        {
            return input.wordTag.Id;
        }

        public string[] Process()
        {
            int j = 0;
            foreach (var input in Input)
            {
                if (input.wordTag.Id == input.resultTag.Id)
                {
                    Result[j] = input.word;
                }
                j++;
            }
            
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
                        
                        var res = proc.Inflect(tpl.input.word, tpl.input.wordTag.Id, tpl.input.resultTag.Id);
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
                var netItems = netTask.Select(tpl => (tpl.input.word, tpl.input.wordTag.Id, tpl.input.resultTag.Id));
                foreach (var netRes in Net.Inflect(netItems))
                {
                    if (Result[netTask[i].index] != null)
                    {
                        i++;
                        continue;
                    }
                    
                    var lexeme = CorrectionDict.Lexeme(netRes.srcWord, netRes.srcTagId);
                    var corResult = lexeme?.FirstOrDefault(w => w.TagId == netRes.resTagId)?.Text;
                    Result[netTask[i].index] = corResult ?? netRes.resWord;
                    i++;
                }
            }
            
            return Result;
        }
    }
}