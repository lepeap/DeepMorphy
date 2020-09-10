using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;

namespace DeepMorphy
{
    internal class GramInfo
    {
        static GramInfo()
        {
            var enRuDic = new Dictionary<string, string>();
            var gramsDic = new Dictionary<string, GramInfo>();
            var gList = new List<GramInfo>();
            using (Stream stream = Utils.GetResourceStream("DeepMorphy.grams.xml"))
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                
                List<Cls> clsList = null;
                while (rdr.Read())
                {
                    
                    if (rdr.Name == "G" && rdr.NodeType == XmlNodeType.Element)
                    {
                        if (gList.Any())
                            gList.Last().Classes = clsList.ToArray();
                        
                        var gr = new GramInfo()
                        {
                            Index = int.Parse(rdr.GetAttribute("index")),
                            KeyEn = rdr.GetAttribute("key_en"),
                            KeyRu = rdr.GetAttribute("key_ru"),
                        };
                        enRuDic[gr.KeyEn] = gr.KeyRu;
                        
                        clsList = new List<Cls>();
                        gList.Add(gr);
                        gramsDic[gr.KeyEn] = gr;
                        gramsDic[gr.KeyRu] = gr;
                    }
                    else  if (rdr.Name == "C" && rdr.NodeType == XmlNodeType.Element)
                    {
                        var cls = new GramInfo.Cls()
                        {
                            NNIndex = rdr.GetAttribute("nn_index") == null
                                ? (long?) null
                                : long.Parse(rdr.GetAttribute("nn_index")),
                            KeyEn = rdr.GetAttribute("key_en"),
                            KeyRu = rdr.GetAttribute("key_ru"),
                        };
                        enRuDic[cls.KeyEn] = cls.KeyRu;
                        clsList.Add(cls);
                    }
                }
                gList.Last().Classes = clsList.ToArray();
            }

            GramsInfo = gList.ToArray();
            EnRuDic = new ReadOnlyDictionary<string, string>(enRuDic);
            RuEnDic = new ReadOnlyDictionary<string, string>(EnRuDic.ToDictionary(x => x.Value, x => x.Key));
            GramsDic = new ReadOnlyDictionary<string, GramInfo>(gramsDic);
            GramCatIndexDic = new ReadOnlyDictionary<int, GramInfo>(gList.ToDictionary(x => x.Index, x => x));
        }

        public int Index { get; private set; }
        public string KeyEn { get; private set; }
        public string KeyRu { get; private set; }
        public Cls[] Classes { get; private set; }
        
        private int? _nnClassesCount;
        public int NnClassesCount
        {
            get
            {
                if (!_nnClassesCount.HasValue)
                    _nnClassesCount = Classes.Count(x => x.NNIndex.HasValue);

                return _nnClassesCount.Value;
            }
        }

        private Dictionary<long, Cls> _nnDict;

        public Cls this[long i]
        {
            get
            {
                if (_nnDict == null)
                    _nnDict = Classes.Where(x => x.NNIndex.HasValue)
                                     .ToDictionary(x => x.NNIndex.Value, x => x);

                return _nnDict[i];
            }
        }
        
        public static GramInfo[] GramsInfo { get; }
        public static ReadOnlyDictionary<string, string> EnRuDic { get; }
        public static ReadOnlyDictionary<string, string> RuEnDic { get; }
        public static ReadOnlyDictionary<string, GramInfo> GramsDic { get;  }
        public static ReadOnlyDictionary<int, GramInfo> GramCatIndexDic { get; }
        
        public static string TranslateKeyToEn(string key)
        {
            return RuEnDic[key];
        }

        public static string TranslateKeyToRu(string key)
        {
            return EnRuDic[key];
        }
        
        public class Cls
        {
            public long? NNIndex { get; set; }
            public string KeyEn { get; set; }
            public string KeyRu { get; set; }
        }
    }
}