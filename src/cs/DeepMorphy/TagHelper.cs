using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;
using DeepMorphy.Exception;

namespace DeepMorphy
{
    public class TagHelper
    {
        internal static int[] LemmasIds { get; }

        internal static Dictionary<int, ReadOnlyDictionary<string, string>> TagsRuDic { get; }
            = new Dictionary<int, ReadOnlyDictionary<string, string>>();

        internal static Dictionary<int, ReadOnlyDictionary<string, string>> TagsEnDic { get; }
            = new Dictionary<int, ReadOnlyDictionary<string, string>>();

        internal static Dictionary<int, string> TagProcDic { get; } = new Dictionary<int, string>();

        internal static Dictionary<int, int> TagOrderDic { get; } = new Dictionary<int, int>();

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
                        var keysEn = keysStr.Split(',');
                        var keysRu = keysEn.Select(x => string.IsNullOrWhiteSpace(x) ? x : GramInfo.EnRuDic[x])
                            .ToArray();

                        var gramDicEn = keysEn.Select((val, i) => (gram: val, index: i))
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

                        if (rdr.GetAttribute("l") != null)
                        {
                            lemmasList.Add(index);
                        }

                        TagsRuDic[index] = new ReadOnlyDictionary<string, string>(gramDicRu);
                        TagsEnDic[index] = new ReadOnlyDictionary<string, string>(gramDicEn);
                        TagProcDic[index] = rdr.GetAttribute("p");
                        TagOrderDic[index] = int.Parse(rdr.GetAttribute("o"));
                    }
                }
            }

            LemmasIds = lemmasList.ToArray();
        }
        
        private readonly bool _useEn;
        private readonly string _postKey = "post";
        private readonly string _nmbrKey = "nmbr";
        private readonly string _gndrKey = "gndr";
        private readonly string _caseKey = "case";
        private readonly string _persKey = "pers";
        private readonly string _tensKey = "tens";
        private readonly string _moodKey = "mood";
        private readonly string _voicKey = "voic";
        
        internal TagHelper(bool useEn)
        {
            _useEn = useEn;
            if (!useEn)
            {
                _postKey = GramInfo.TranslateKeyToRu(_postKey);
                _nmbrKey = GramInfo.TranslateKeyToRu(_nmbrKey);
                _gndrKey = GramInfo.TranslateKeyToRu(_gndrKey);
                _caseKey = GramInfo.TranslateKeyToRu(_caseKey);
                _persKey = GramInfo.TranslateKeyToRu(_persKey);
                _tensKey = GramInfo.TranslateKeyToRu(_tensKey);
                _moodKey = GramInfo.TranslateKeyToRu(_moodKey);
                _voicKey = GramInfo.TranslateKeyToRu(_voicKey);
            }

            TagsDic = useEn ? TagsEnDic : TagsRuDic;
        }

        internal static bool IsLemma(int tagId)
        {
            return LemmasIds.Contains(tagId);
        }

        internal Dictionary<int, ReadOnlyDictionary<string, string>> TagsDic { get; }

        internal Tag CreateTagFromId(int tagId, float power = 1.0f, string lemma = null)
        {
            return new Tag(TagsDic[tagId], power, tagId, lemma);
        }
        
        public Tag CreateTag(string post,
                             string gndr = null,
                             string nmbr = null,
                             string @case = null,
                             string pers = null,
                             string tens = null,
                             string mood = null,
                             string voic = null)
        {
            var foundTags = FilterTags(post, gndr, nmbr, @case, pers, tens, mood, voic, fullMatch: true).ToArray();
            if (foundTags.Length > 1)
            {
                var tags = foundTags.Select(t=>t.ToString());
                var tagsText = string.Join("\n", tags);
                var message = _useEn
                    ? $"Ambigious gram values. Found several possible tags:\n{tagsText}"
                    : $"Неоднозначные значения граммем. Найдено несколько допустимых тэгов:\n{tagsText}";
                throw new AmbigGramsForTagException(message);
            }

            if (foundTags.Length == 0)
            {
                var message = _useEn
                    ? "Tag not found"
                    : "Тег не найден";
                throw new TagNotSupportedException(message);
            }

            return foundTags[0];
        }

        public IEnumerable<Tag> FilterTags(string post,
                                           string gndr = null,
                                           string nmbr = null,
                                           string @case = null,
                                           string pers = null,
                                           string tens = null,
                                           string mood = null,
                                           string voic = null,
                                           bool fullMatch = false)
        {
            return TagsDic.Where(t =>
            {
                if (t.Value[_postKey] != post)
                {
                    return false;
                }

                var tGndr = t.Value.ContainsKey(_gndrKey) ? t.Value[_gndrKey] : null;
                var fm = fullMatch ? true : gndr != null;
                if (fm && tGndr != gndr)
                {
                    return false;
                }

                var tNmbr = t.Value.ContainsKey(_nmbrKey) ? t.Value[_nmbrKey] : null;
                fm = fullMatch ? true : nmbr != null;
                if (fm && tNmbr != nmbr)
                {
                    return false;
                }

                var tCase = t.Value.ContainsKey(_caseKey) ? t.Value[_caseKey] : null;
                fm = fullMatch ? true : @case != null;
                if (fm && tCase != @case)
                {
                    return false;
                }

                var tPers = t.Value.ContainsKey(_persKey) ? t.Value[_persKey] : null;
                fm = fullMatch ? true : pers != null;
                if (fm && tPers != pers)
                {
                    return false;
                }

                var tTens = t.Value.ContainsKey(_tensKey) ? t.Value[_tensKey] : null;
                fm = fullMatch ? true : tens != null;
                if (fm && tTens != tens)
                {
                    return false;
                }

                var tMood = t.Value.ContainsKey(_moodKey) ? t.Value[_moodKey] : null;
                fm = fullMatch ? true : mood != null;
                if (fm && tMood != mood)
                {
                    return false;
                }

                var tVoid = t.Value.ContainsKey(_voicKey) ? t.Value[_voicKey] : null;
                fm = fullMatch ? true : tens != null;
                if (fm && tVoid != voic)
                {
                    return false;
                }
                return true;
            })
            .Select(kp => new Tag(kp.Value, 1, kp.Key))
            .OrderByDescending(t => t.Id);
        }

        public IEnumerable<Tag> ListSupportedTags()
        {
            return TagsDic.OrderByDescending(kp => TagOrderDic[kp.Key])
                          .Select(kp => new Tag(kp.Value, 1, kp.Key));
        }

        public ReadOnlyDictionary<string, string> this[int tagIndex]
        {
            get => TagsDic[tagIndex];
        }
    }
}