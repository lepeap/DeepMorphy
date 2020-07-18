namespace DeepMorphy
{
    public class GramHelper
    {
        public string TranslateKeyToEn(string key)
        {
            return GramInfo.RuEnDic[key];
        }

        public string TranslateKeyToRu(string key)
        {
            return GramInfo.EnRuDic[key];
        }
    }
}