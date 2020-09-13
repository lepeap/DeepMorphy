using System.Collections.Generic;
using System.Linq;
using DeepMorphy.Model;

namespace DeepMorphy.Split
{
    internal class InflectProc : SplittedProc<InflectTask, string>
    {
        public InflectProc(IEnumerable<InflectTask> input, MorphAnalyzer morph)
            : base(input, morph)
        {
        }

        protected override int _GetTagIndex(InflectTask input)
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

            _processNetworkItems();
            return Result;
        }

        private void _processNetworkItems()
        {
            List<(InflectTask input, int index)> netTask = new List<(InflectTask task, int index)>();
            foreach (var task in GetForProcessor("nn"))
            {
                if (TagHelper.TagProcDic[task.input.resultTag.Id] != "nn")
                {
                    Result[task.index] = null;
                    continue;
                }
                netTask.Add(task);
            }

            if (netTask.Count != 0)
            {
                var lemTask = netTask.Select(x => new LemTask(x.input.word, x.input.wordTag));
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
                        input: new InflectTask(lemResult, lemTag, srcInput.resultTag),
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
        }
    }
}