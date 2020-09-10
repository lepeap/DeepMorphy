using System.Collections.Generic;
using System.Linq;

namespace DeepMorphy.Split
{
    internal class InflectProc : SplittedProc<(string word, Tag wordTag, Tag resultTag), string>
    {
        public InflectProc(IEnumerable<(string word, Tag wordTag, Tag resultTag)> input, MorphAnalyzer morph)
            : base(input, morph)
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
                var lemTask = netTask.Select(x => (x.input.word, x.input.wordTag));
                var lemResults = Morph.Lemmatize(lemTask);
                int i = 0;
                foreach (var lemResult in lemResults)
                {
                    var srcTask = netTask[i];
                    var srcInput = srcTask.input;
                    var lemTagId = Morph.Net.GetLemmaTagId(srcInput.wordTag.Id);
                    Tag lemTag = lemTagId == srcTask.input.wordTag.Id
                        ? srcInput.wordTag
                        : Morph.TagHelper.CreateTagFromId(lemTagId);
                    netTask[i] = (
                        input: (
                            word: lemResult, 
                            wordTag: lemTag,
                            resultTag: srcInput.resultTag
                        ),
                        index: srcTask.index
                    );
                    i++;
                }

                i = 0;
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