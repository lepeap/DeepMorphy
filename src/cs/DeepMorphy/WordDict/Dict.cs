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
        
        public IEnumerable<Word> Parse(string word)
        {
            var isInDic = _indexDic.TryGetValue(word, out int[] lexemes);
            if (!isInDic)
            {
                yield break;
            }

            foreach (var id in lexemes)
            {
                var lexemeWords = _parseLexeme(id).ToArray();
                string lemma = null;// lexemeWords.First(x => TagHelper.IsLemma(x.TagId)).Text;
                foreach (var dWord in lexemeWords)
                {
                    if (dWord.Text != word)
                    {
                        continue;
                    }

                    dWord.Lemma = lemma;
                }
            }
        }
        
        public Word[] Lexeme(string word, int tagId)
        {
            var isInDic = _indexDic.TryGetValue(word, out int[] lexemes);
            if (!isInDic)
            {
                return null;
            }

            foreach (var id in lexemes)
            {
                var lexeme = _parseLexeme(id).ToArray();
                if (lexeme.Any(x => x.Text == word && x.TagId == tagId))
                {
                    return lexeme;
                }
            }

            return null;
        }
        
        private IEnumerable<Word> _parseLexeme(int lexemeId)
        {
            var srcVal = _lexemeDic[lexemeId];
            foreach (var formVal in srcVal.Split(';'))
            {
                var splMas = formVal.Split(':');
                string cls;
                string word = splMas[0];
                foreach (var clsVal in splMas[1].Split(','))
                {
                    bool replaceOther;
                    if (clsVal.EndsWith("!"))
                    {
                        cls = clsVal.Substring(0, clsVal.Length - 1);
                        replaceOther = true;
                    }
                    else
                    {
                        cls = clsVal;
                        replaceOther = false;
                    }
                
                    yield return new Word(word, int.Parse(cls), replaceOther);
                }
            }
        }
    }
}