using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;

namespace DeepMorphy.WordDict
{
    class Dict
    {
        
        private static readonly char[] CommmaSplitDict = new[] {','};
        private readonly Dictionary<string, string> _gramDic = new Dictionary<string, string>();
        private readonly Dictionary<string, string> _index = new Dictionary<string, string>();
        private readonly bool _withLemmatization;
        private readonly bool _useEnGrams;
        public Dict(bool useEnGrams, bool withLemmatization)
        {
            _withLemmatization = withLemmatization;
            _useEnGrams = useEnGrams;
            using (var reader = new StreamReader(_getTxtStream(), Encoding.UTF8))
            {
                var line = reader.ReadLine();
                while (!string.IsNullOrWhiteSpace(line))
                {
                    var spltRez = line.Split('=');
                    _gramDic[spltRez[0]] = spltRez[1];
                    line = reader.ReadLine();
                }

                while (!reader.EndOfStream)
                {
                    line = reader.ReadLine();
                    var spltRez = line.Split('\t');
                    _index[spltRez[0]] = spltRez[1];
                }
            }
        }
        private Stream _getTxtStream()
        {
            var resourceName = $"DeepMorphy.WordDict.dict.txt.gz";
            return Utils.GetCompressedResourceStream(resourceName);
        }
        
        public MorphInfo Parse(string word)
        {
            if (_index.ContainsKey(word))
            {
                var srcTags = _index[word];
                var tags = _parseTags(word, srcTags).ToArray();            
                return new MorphInfo(word, tags, _useEnGrams);
            }

            return null;
        }

        private IEnumerable<Tag> _parseTags(string word, string srcTags)
        {
            var tagsSrcs = srcTags.Split(';').ToArray();
            foreach (var form in tagsSrcs)
            {
                var splRez = form.Split(':');
                var gramDic = splRez[1]
                                .Split(',')
                                .Select((val, i) => (
                                    gram: string.IsNullOrEmpty(val) ? string.Empty : _gramDic[val], 
                                    index: i
                                ))
                                .Where(tpl => !string.IsNullOrEmpty(tpl.gram))
                                .ToDictionary(
                                    x => _useEnGrams 
                                        ? GramInfo.GramCatIndexDic[x.index].KeyEn 
                                        : GramInfo.GramCatIndexDic[x.index].KeyRu,
                                    x => _useEnGrams ? x.gram : GramInfo.EnRuDic[x.gram]
                                );
                string lemma = null;
                if (_withLemmatization && !string.IsNullOrWhiteSpace(splRez[0]))
                    lemma = splRez[0];    
                else if (_withLemmatization)
                    lemma = word;

                yield return new Tag(
                    new ReadOnlyDictionary<string, string>(gramDic),
                    (float) 1.0 / tagsSrcs.Length,
                    lemma
                );
            }
        }
    }
}