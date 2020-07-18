using System.Linq;
using DeepMorphy.NeuralNet;

namespace DeepMorphy
{
    public class TagHelper
    {
        private readonly MorphAnalyzer _morph;
        private readonly Config _networkConfig;
        
        private readonly string _postKey;
        private readonly string _numberKey;
        private readonly string _genderKey;
        private readonly string _caseKey;
        
        private readonly string _infnKey;
        private readonly string _nounKey;
        internal TagHelper(MorphAnalyzer morph, Config networkConfig)
        {
            _morph = morph;
            _networkConfig = networkConfig;
            _postKey = "post";
            _numberKey = "nmbr";
            _genderKey = "gndr";
            _caseKey = "case";
            
            _infnKey = "infn";
            _nounKey = "noun";
            

            if (!morph.EnTags)
            {
                var helper = morph.GramHelper;
                _postKey = helper.TranslateKeyToRu(_postKey);
                _numberKey = helper.TranslateKeyToRu(_numberKey);
                _genderKey = helper.TranslateKeyToRu(_genderKey);
                _caseKey = helper.TranslateKeyToRu(_caseKey);
                
                _infnKey = helper.TranslateKeyToRu(_infnKey);
                _nounKey = helper.TranslateKeyToRu(_nounKey);
            }

        }
        
        public Tag CreateForInfn(string word)
        {
            var keyValuePair = _networkConfig.ClsDic
                                             .Single(x => x.Value[_postKey] == _infnKey);
            return new Tag(keyValuePair.Value, 1, word, keyValuePair.Key);
        }

        public Tag CreateForNoun(string word, string number, string gender, string @case, string lemma=null)
        {
            var keyValuePair = _networkConfig.ClsDic
                                             .Single(x => x.Value[_postKey] == _nounKey
                                                                 && x.Value.ContainsKey(_numberKey) 
                                                                 && x.Value[_numberKey] == number
                                                                 && x.Value.ContainsKey(_genderKey) 
                                                                 && x.Value[_genderKey] == gender
                                                                 && x.Value.ContainsKey(_caseKey) 
                                                                 && x.Value[_caseKey] == @case);
            var tag = new Tag(keyValuePair.Value, 1, word, keyValuePair.Key);
            if (lemma == null)
            {
                lemma = _morph.Lemmatize(word, tag);
            }

            tag.Lemma = lemma;
            return tag;
        }
    }
}