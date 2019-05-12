using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Xml;

namespace DeepMorphy.NeuralNet
{
    class Config
    {
        private readonly char[] _commmaSplitDict = {','};

        public Config(bool useEnTags, bool bigModel)
        {
            UseEnTags = useEnTags;
            BigModel = bigModel;
            _loadReleaseInfo();
        }
        
        public bool UseEnTags { get; private set; }
        public bool BigModel { get; private set; }
        public int UndefinedCharId { get; private set; }
        public int StartCharIndex { get; private set; }
        public int EndCharIndex { get; private set; }
        
        public int MainClassK { get; private set; }
        public Dictionary<int, string[]> ClsDic { get; private set; } = new Dictionary<int, string[]>();
        public Dictionary<char, int> CharToId { get; private set; } = new Dictionary<char, int>();
        
        public Dictionary<int, char> IdToChar { get; private set; } = new Dictionary<int, char>();
        public Dictionary<string, string> OpDic { get; private set; } = new Dictionary<string, string>();
        public  Dictionary<string, string>  GramOpDic { get; private set; } = new Dictionary<string, string>();

        private void _loadReleaseInfo()
        {
            using (Stream stream = _getXmlStream(BigModel))
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
                        if (!UseEnTags)
                            key = Gram.EnRuDic[key];

                        GramOpDic[key] = rdr.GetAttribute("op");
                    }
                    else if (rdr.Name.Equals("C") && rdr.NodeType == XmlNodeType.Element)
                    {
                        var index = int.Parse(rdr.GetAttribute("i"));
                        var keysStr = rdr.GetAttribute("v");
                        var keys = keysStr.Split(_commmaSplitDict, StringSplitOptions.RemoveEmptyEntries);

                        if (!UseEnTags)
                            keys = keys.Select(x => Gram.EnRuDic[x]).ToArray();
                        
                        ClsDic[index] = keys;
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
                }
            }
        }

        private Stream _getXmlStream(bool bigModel)
        {
            var modelKey = bigModel ? "big" : "small";
            var resourceName = $"DeepMorphy.NeuralNet.release_{modelKey}.xml";
            return Utils.GetResourceStream(resourceName);
        }


        public Gram this[string gramKey] => Gram.GramsDic[gramKey];

        public string this[string gramKey, long i]
        {
            get
            {
                var cls = Gram.GramsDic[gramKey][i];
                return UseEnTags ? cls.KeyEn : cls.KeyRu;
            }
        }
    }
}