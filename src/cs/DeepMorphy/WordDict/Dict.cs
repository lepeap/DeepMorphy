using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DeepMorphy.WordDict
{
    internal class Dict
    {
        private readonly SortedDictionary<string, int[]> _indexDic = new SortedDictionary<string, int[]>();
        private readonly SortedDictionary<int, string> _lexemeDic = new SortedDictionary<int, string>();

        public Dict()
        {
            
        }
        
        public Dict(string dictKey)
        {
            using (var reader = new StreamReader(Utils.GetCompressedResourceStream($"DeepMorphy.WordDict.{dictKey}_index.txt.gz"), Encoding.UTF8))
            {
                var line = reader.ReadLine();
                while(true)
                {
                    var spltRez = line.Split(':');
                    _indexDic[spltRez[0]] = spltRez[1].Split(',').Select(x => int.Parse(x)).ToArray();
                    if (reader.EndOfStream)
                    {
                        break;
                    }
                    
                    line = reader.ReadLine();
                }
            }
            
            using (var reader = new StreamReader(Utils.GetCompressedResourceStream($"DeepMorphy.WordDict.{dictKey}.txt.gz"), Encoding.UTF8))
            {
                var line = reader.ReadLine();
                while(true)
                {
                    var spltRez = line.Split('\t');
                    int lexemeId = int.Parse(spltRez[0]);
                    _lexemeDic[lexemeId] = spltRez[1];
                    if (reader.EndOfStream)
                    {
                        break;
                    }
                    
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
                string lemma = lexemeWords.FirstOrDefault(x => TagHelper.IsLemma(x.TagId))?.Text;
                if (lemma == null)
                {
                    lemma = lexemeWords[0].Text;
                }
                
                foreach (var dWord in lexemeWords)
                {
                    if (dWord.Text != word)
                    {
                        continue;
                    }

                    dWord.Lemma = lemma;
                    yield return dWord;
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
                
                string word = splMas[0];
                var splRez = splMas[1].Split(',');
                foreach (var clsVal in splRez)
                {
                    string cls;
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