using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml;

namespace DeepMorphy.Numb
{
    internal static class NumbInfo
    {
        static NumbInfo()
        {
            using (Stream stream = Utils.GetResourceStream("DeepMorphy.Numb.numbers.xml"))
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                NumberData currentVal = null;
                while (rdr.Read())
                {
                    if (rdr.Name == "N" && rdr.NodeType == XmlNodeType.Element)
                    {
                        var val = rdr.GetAttribute("v");
                        currentVal = new NumberData();
                        NumberDictionary[val] = currentVal;
                    }
                    else if (rdr.Name == "W" && rdr.NodeType == XmlNodeType.Element && rdr.GetAttribute("p") != null)
                    {
                        var clsId = int.Parse(rdr.GetAttribute("i"));
                        var text = rdr.GetAttribute("t");
                        currentVal.Ordinal[clsId] = text;
                    }
                    else if (rdr.Name == "W" && rdr.NodeType == XmlNodeType.Element)
                    {
                        var clsId = int.Parse(rdr.GetAttribute("i"));
                        var text = rdr.GetAttribute("t");
                        currentVal.Quantitative[clsId] = text;
                    }
                    else if (rdr.Name == "E" && rdr.NodeType == XmlNodeType.Element)
                    {
                        var clsId = int.Parse(rdr.GetAttribute("i"));
                        var text = rdr.GetAttribute("t");
                        currentVal.NarEnd[clsId] = text;
                    }
                    else if (rdr.Name == "NumbData" && rdr.NodeType == XmlNodeType.Element)
                    {
                        var reg = rdr.GetAttribute("reg");
                        NumberRegex = new Regex(reg, RegexOptions.Compiled);
                        LemmaTagId = int.Parse(rdr.GetAttribute("l"));
                        RegexGroups = NumberRegex.GetGroupNames()
                                                 .Where(x => x.StartsWith("_"))
                                                 .ToDictionary(x => x,x => x.Substring(1, x.Length -1));
                    }
                }
            }
        }

        public static int LemmaTagId { get; }

        public static Regex NumberRegex { get; }
        
        public static Dictionary<string, string> RegexGroups { get; }

        public static Dictionary<string, NumberData> NumberDictionary { get; } = new Dictionary<string, NumberData>();

        internal class NumberData
        {
            public Dictionary<int, string> Ordinal = new Dictionary<int, string>();

            public Dictionary<int, string> Quantitative = new Dictionary<int, string>();

            public Dictionary<int, string> NarEnd = new Dictionary<int, string>();
        }
    }
}