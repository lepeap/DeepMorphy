using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;
using DeepMorphy;
using Newtonsoft.Json;

namespace AmbDataset
{
    class Program
    {
        static void Main(string[] args)
        {
            var resPath = "/media/alex/hybrid/Projects/DeepMorphy/ambig_src.json";
            var srcFilePath = @"/media/alex/hybrid/Projects/Resources/annot.opcorpora.no_ambig.nonmod.xml";
            var sents = LoadSents(srcFilePath);
            ParseTokens(sents);
            File.WriteAllText(resPath, JsonConvert.SerializeObject(sents));
        }
        
        static List<List<Token>> LoadSents(string srcFilePath)
        {
            using (Stream stream = File.Open(srcFilePath, FileMode.Open))
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                List<List<Token>> resList = new List<List<Token>>();
                List<Token> curSent = null;
                Token token = null;
                while (rdr.Read())
                {
                    if (rdr.Name == "tokens" && rdr.NodeType == XmlNodeType.Element)
                    {
                        curSent = new List<Token>();
                        resList.Add(curSent);
                    }
                    else  if (rdr.Name == "token" && rdr.NodeType == XmlNodeType.Element)
                    {
                        token = new Token()
                        {
                            Text = rdr.GetAttribute("text")
                        };
                        curSent.Add(token);
                    }
                    else if (rdr.Name == "g" && rdr.NodeType == XmlNodeType.Element)
                    {
                        token.Grams.Add(rdr.GetAttribute("v").ToLower());
                    }
                }

                return resList;
            }
        }

        static void ParseTokens(List<List<Token>> tokens)
        {
            var morph = new MorphAnalyzer(useEnGrams: true);
            foreach (var sentTokens in tokens)
            {
                var infs = morph.Parse(sentTokens.Select(x => x.Text)).ToArray();
                for (int i = 0; i < infs.Length; i++)
                {
                    sentTokens[i].Tags = infs[i].Tags.Where(x => TagHelper.TagProcDic[x.Id] != "nn").Select(x => x.Id).ToArray();
                }
            }
        }
    }
}