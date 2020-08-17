using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DeepMorphy.WordDict
{
    internal class Dict
    {
        private readonly Dictionary<string, int[]> _indexDic = new Dictionary<string, int[]>();
        private readonly Dictionary<int, string> _lexemeDic = new Dictionary<int, string>();

        public Dict(string dictKey)
        {
            using (var reader = new StreamReader(Utils.GetCompressedResourceStream($"DeepMorphy.WordDict.{dictKey}_index.txt.gz"), Encoding.UTF8))
            {
                var line = reader.ReadLine();
                while (!reader.EndOfStream && !string.IsNullOrWhiteSpace(line))
                {
                    var spltRez = line.Split(':');
                    _indexDic[spltRez[0]] = spltRez[1].Split(',').Select(x => int.Parse(x)).ToArray();
                    line = reader.ReadLine();
                }
            }
            
            using (var reader = new StreamReader(Utils.GetCompressedResourceStream($"DeepMorphy.WordDict.{dictKey}.txt.gz"), Encoding.UTF8))
            {
                var line = reader.ReadLine();
                while (!reader.EndOfStream && !string.IsNullOrWhiteSpace(line))
                {
                    var spltRez = line.Split('\t');
                    _lexemeDic[int.Parse(spltRez[0])] = spltRez[1];
                    line = reader.ReadLine();
                }
            }
        }
        
        public IEnumerable<IEnumerable<(string word, int tag)>> Get(string word)
        {
            var isInDic = _indexDic.TryGetValue(word, out int[] forms);
            if (!isInDic)
            {
                yield break;
            }

            foreach (var formId in forms)
            {
                yield return _parseLexeme(formId);
            }
        }

        private IEnumerable<(string word, int tag)> _parseLexeme(int lexemeId)
        {
            var srcVal = _lexemeDic[lexemeId];
            foreach (var formVal in srcVal.Split(';'))
            {
                var splMas = formVal.Split(':');
                yield return (splMas[0], int.Parse(splMas[1]));
            }
        }
    }
}