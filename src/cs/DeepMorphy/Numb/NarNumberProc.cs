using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace DeepMorphy.Numb
{
    internal class NarNumberProc : IMorphProcessor
    {
        private static readonly Regex NarChislReg = new Regex(@"(\d+)-([а-я]+)", RegexOptions.Compiled);
        private static readonly char[] GlasnChars = {'а', 'о', 'и', 'е', 'ё', 'э', 'ы', 'у', 'ю', 'я'};
        private static readonly char[] SoglChars = {'б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ'};
        private static readonly Dictionary<string, string> Templates = new Dictionary<string, string>()
        {
            {"11", "11"},
            {"12", "12"},
            {"13", "13"},
            {"14", "14"},
            {"15", "15"},
            {"16", "16"},
            {"17", "17"},
            {"18", "18"},
            {"19", "19"},
            {"1", "1"},
            {"2", "2"},
            {"3", "3"},
            {"4", "4"},
            {"5", "5"},
            {"6", "6"},
            {"7", "7"},
            {"8", "8"},
            {"9", "9"},
            {"20", "20"},
            {"30", "30"},
            {"40", "40"},
            {"50", "50"},
            {"60", "60"},
            {"70", "70"},
            {"80", "80"},
            {"90", "90"},
            {"100", "100"},
            {"200", "200"},
            {"300", "300"},
            {"400", "400"},
            {"500", "500"},
            {"600", "600"},
            {"700", "700"},
            {"800", "800"},
            {"900", "900"},
            {"1000", "1000"},
            {"10000", "1000"},
            {"100000", "1000"},
            {"1000000", "1000000"},
            {"10000000", "1000000"},
            {"100000000", "1000000"},
            {"1000000000", "1000000000"},
            {"10000000000", "1000000000"},
            {"100000000000", "1000000000"},
            {"0", "0"},
        };

        public string Key => "numb";

        public bool IgnoreNetworkResult => false;

        public IEnumerable<(int tagId, string lemma)> Parse(string word)
        {
            var data = _tryParse(word);
            if (data == null)
            {
                return null;
            }

            var numbData = data.NumberData;
            var lemmaEnd = numbData.NarEnd.First(x => NumbInfo.LemmaTagId.Contains(x.Key)).Value;
            var lemma = $"{data.Digit}-{lemmaEnd}";
            return numbData.Lexemes["p"].Where(tpl => tpl.text.EndsWith(data.OriginalEnd)).Select(tpl => (tpl.tagId, lemma));
        }

        public string Lemmatize(string word, int tagId)
        {
            var data = _tryParse(word);
            if (data == null)
            {
                return null;
            }
            
            var numbData = data.NumberData;
            var lemmaEnd = numbData.NarEnd.First(x => NumbInfo.LemmaTagId.Contains(x.Key)).Value;
            var lemma = $"{data.Digit}-{lemmaEnd}";
            return lemma;
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            var data = _tryParse(word);
            if (data == null)
            {
                return null;
            }

            var numbData = data.NumberData;
            if (!numbData.NarEnd.ContainsKey(resultTag))
            {
                return null;
            }

            var result = $"{data.Digit}-{numbData.NarEnd[resultTag]}";
            return result;
        }

        public IEnumerable<(int tagId, string text)> Lexeme(string word, int tag)
        {
            var data = _tryParse(word);
            if (data == null)
            {
                return null;
            }

            var numbData = data?.NumberData;
            return numbData?.NarEnd.Select(kp => (kp.Key, $"{data.Digit}-{kp.Value}"));
        }

        private NarNumberData _tryParse(string word)
        {
            var match = NarChislReg.Match(word);
            if (!match.Success)
            {
                return null;
            }

            var digit = match.Groups[1].Value;
            var end = match.Groups[2].Value;
            var correctEnd = _correctEnd(end);
            var numbKey = Templates.Keys.First(x => digit.EndsWith(x));
            var numbData = NumbInfo.NumberDictionary[numbKey];
            return new NarNumberData()
            {
                OriginalEnd = end,
                End = correctEnd,
                Digit = digit,
                NumberData = numbData
            };
        }

        private string _correctEnd(string val)
        {
            if (val.Length == 1)
                return val;

            if (GlasnChars.Contains(val[val.Length - 1]) && SoglChars.Contains(val[val.Length - 2]))
                return val.Substring(val.Length - 2);

            return val[val.Length - 1].ToString();
        }

        private class NarNumberData
        {
            public string Digit { get; set; }

            public string OriginalEnd { get; set; }

            public string End { get; set; }

            public NumbInfo.NumberData NumberData { get; set; }
        }
    }
}