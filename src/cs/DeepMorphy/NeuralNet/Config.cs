using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;

namespace DeepMorphy.NeuralNet
{
    internal class Config
    {
        public Config(bool useEnGrams)
        {
            UseEnGrams = useEnGrams;
            _loadReleaseInfo();
        }
        
        public bool UseEnGrams { get; }
        public int UndefinedCharId { get; private set; }
        public int StartCharIndex { get; private set; }
        public int EndCharIndex { get; private set; }
        public ReadOnlyDictionary<int, int[]> InflectTemplatesDic { get; private set;  }
        public ReadOnlyDictionary<int, int> ClsToLemmaDic { get; private set;  }
        public Dictionary<char, int> CharToId { get; } = new Dictionary<char, int>();
        public Dictionary<int, char> IdToChar { get; } = new Dictionary<int, char>();
        public Dictionary<string, string> OpDic { get; } = new Dictionary<string, string>();
        public  Dictionary<string, string>  GramOpDic { get; } = new Dictionary<string, string>();
        public GramInfo this[string gramKey] => GramInfo.GramsDic[gramKey];
        
        public string this[string gramKey, long i]
        {
            get
            {
                var cls = GramInfo.GramsDic[gramKey][i];
                return UseEnGrams ? cls.KeyEn : cls.KeyRu;
            }
        }

        private void _loadReleaseInfo()
        {
            var inflectDic = new Dictionary<int, List<int>>();
            var clsToLemmaDic = new Dictionary<int, int>();
            int curInflectLemmaId = -1;

            using (Stream stream = _getXmlStream())
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                while (rdr.Read())
                {
                    if (rdr.Name == "Char" && rdr.NodeType == XmlNodeType.Element)
                    {
                        string val = rdr.GetAttribute("value");
                        int index = int.Parse(rdr.GetAttribute("index"));
                        if (val == "UNDEFINED")
                            UndefinedCharId = index;
                        else
                        {
                            CharToId[val[0]] = index;
                            IdToChar[index] = val[0];
                        }
                    }
                    else if (rdr.Name.Equals("G") && rdr.NodeType == XmlNodeType.Element)
                    {
                        var key = rdr.GetAttribute("key");
                        if (!UseEnGrams)
                            key = GramInfo.EnRuDic[key];

                        GramOpDic[key] = rdr.GetAttribute("op");
                    }
                    
                    else if (rdr.Name.Equals("Chars") && rdr.NodeType == XmlNodeType.Element)
                    {
                        StartCharIndex =  int.Parse(rdr.GetAttribute("start_char"));
                        EndCharIndex =  int.Parse(rdr.GetAttribute("end_char"));
                    }
                    else if (rdr.Name.Equals("Root") && rdr.NodeType == XmlNodeType.Element)
                    {
                        rdr.MoveToFirstAttribute();
                        OpDic[rdr.Name] = rdr.Value;
                        
                        while (rdr.MoveToNextAttribute())
                            OpDic[rdr.Name] = rdr.Value;
        
                        rdr.MoveToElement();
                    }
                    
                    else if (rdr.Name == "Im" && rdr.NodeType == XmlNodeType.Element)
                    {
                        curInflectLemmaId = int.Parse(rdr.GetAttribute("i"));
                        inflectDic[curInflectLemmaId] = new List<int>();
                    }
                    else if (rdr.Name == "I" && rdr.NodeType == XmlNodeType.Element)
                    {
                        var clsIndex = int.Parse(rdr.GetAttribute("i"));
                        inflectDic[curInflectLemmaId].Add(clsIndex);
                        clsToLemmaDic[clsIndex] = curInflectLemmaId;
                    }
                }
            }
            InflectTemplatesDic =
                new ReadOnlyDictionary<int, int[]>(inflectDic.ToDictionary(x => x.Key, x => x.Value.ToArray()));
            ClsToLemmaDic = new ReadOnlyDictionary<int, int>(clsToLemmaDic);
        }

        private Stream _getXmlStream()
        {
            var resourceName = $"DeepMorphy.NeuralNet.release_small.xml";
            return Utils.GetResourceStream(resourceName);
        }
    }
}