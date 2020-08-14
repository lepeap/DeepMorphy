using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;
using DeepMorphy.NeuralNet;

namespace DeepMorphy
{
    public class TagHelper
    {
        private static readonly char[] _commmaSplitter = {','};
        internal static int[] LemmasIds { get; }
        internal static Dictionary<int, ReadOnlyDictionary<string, string>> TagsRuDic  { get; } 
            = new Dictionary<int, ReadOnlyDictionary<string, string>>();
        internal static Dictionary<int, ReadOnlyDictionary<string, string>> TagsEnDic  { get; } 
            = new Dictionary<int, ReadOnlyDictionary<string, string>>();
        internal static Dictionary<int, string> TagProcDic { get; } = new Dictionary<int, string>();
        
        static TagHelper()
        {
            var lemmasList = new List<int>();
            using (Stream stream = Utils.GetResourceStream("DeepMorphy.tags.xml"))
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                while (rdr.Read())
                {
                    if (rdr.Name.Equals("T") && rdr.NodeType == XmlNodeType.Element)
                    {
                        var index = int.Parse(rdr.GetAttribute("i"));
                        var keysStr = rdr.GetAttribute("v");
                        var keysEn = keysStr.Split(_commmaSplitter);
                        var keysRu = keysEn.Select(x => string.IsNullOrWhiteSpace(x) ? x : GramInfo.EnRuDic[x]).ToArray();

                        
                        var gramDicEn = keysRu.Select((val, i) => (gram: val, index: i))
                            .Where(tpl => !string.IsNullOrEmpty(tpl.gram))
                            .ToDictionary(
                                x => GramInfo.GramCatIndexDic[x.index].KeyEn,
                                x => x.gram
                            );
                        var gramDicRu = keysRu.Select((val, i) => (gram: val, index: i))
                            .Where(tpl => !string.IsNullOrEmpty(tpl.gram))
                            .ToDictionary(
                                x => GramInfo.GramCatIndexDic[x.index].KeyRu,
                                x => x.gram
                            );
                        
                        if (rdr.GetAttribute("lem") != null)
                        {
                            lemmasList.Add(index);
                        }
                        
                        TagsRuDic[index] = new ReadOnlyDictionary<string, string>(gramDicRu);
                        TagsEnDic[index] = new ReadOnlyDictionary<string, string>(gramDicEn);
                        TagProcDic[index] = rdr.GetAttribute("p");
                    }
                }
            }

            LemmasIds = lemmasList.ToArray();
        }
        
        private readonly MorphAnalyzer _morph;
        private readonly string _postKey;
        private readonly string _nmbrKey;
        private readonly string _genderKey;
        private readonly string _caseKey;
        private readonly string _infnKey;
        
        
        private readonly string _nounKey;
        private readonly string _numberKey;
        
        internal TagHelper(MorphAnalyzer morph)
        {
            _morph = morph;
            _postKey = "post";
            _nmbrKey = "nmbr";
            _genderKey = "gndr";
            _caseKey = "case";
            _infnKey = "infn";
            
            _nounKey = "noun";
            _numberKey = "numb";
            
            if (!morph.EnTags)
            {
                var helper = morph.GramHelper;
                _postKey = helper.TranslateKeyToRu(_postKey);
                _nmbrKey = helper.TranslateKeyToRu(_nmbrKey);
                _genderKey = helper.TranslateKeyToRu(_genderKey);
                _caseKey = helper.TranslateKeyToRu(_caseKey);
                
                _infnKey = helper.TranslateKeyToRu(_infnKey);
                _nounKey = helper.TranslateKeyToRu(_nounKey);
                _numberKey = helper.TranslateKeyToRu(_numberKey);
            }

            TagsDic = morph.EnTags ? TagsEnDic : TagsRuDic;
        }
        
        internal Dictionary<int, ReadOnlyDictionary<string, string>>  TagsDic { get; }

        public Tag CreateForInfn(string word)
        {
            var keyValuePair = TagsDic.Single(x => x.Value[_postKey] == _infnKey);
            return new Tag(keyValuePair.Value, 1, word, keyValuePair.Key);
        }

        public Tag CreateForNoun(string word, string number, string gender, string @case, string lemma=null)
        {
            return _createTag(word,
                lemma,
                x => x.Value[_postKey] == _nounKey
                     && x.Value.ContainsKey(_nmbrKey)
                     && x.Value[_nmbrKey] == number
                     && x.Value.ContainsKey(_genderKey)
                     && x.Value[_genderKey] == gender
                     && x.Value.ContainsKey(_caseKey)
                     && x.Value[_caseKey] == @case
            );
        }

        public Tag CreateForNumb(string word, string gender, string @case, string lemma = null)
        {
            return _createTag(word,
                lemma,
                x => x.Value[_postKey] == _nounKey
                           && x.Value.ContainsKey(_nmbrKey)
                           && x.Value[_genderKey] == gender
                           && x.Value.ContainsKey(_caseKey)
                           && x.Value[_caseKey] == @case
            );
        }

        private Tag _createTag(string word, 
                               string lemma,
                               Func<KeyValuePair<int, ReadOnlyDictionary<string, string>>, bool> filter)
        {
            var keyValuePair = TagsDic.Single(filter);
            var tag = new Tag(keyValuePair.Value, 1, lemma, keyValuePair.Key);
            
            if (lemma == null)
            {
                tag.Lemma = _morph.Lemmatize(word, tag);
            }

            return tag;
        }

        public ReadOnlyDictionary<string, string> this[int tagIndex]
        {
            get => TagsDic[tagIndex];
        }
    }
}