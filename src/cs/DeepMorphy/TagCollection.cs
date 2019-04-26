using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace DeepMorphy
{
    public sealed class TagCollection
    {
        private Tag[] _tags;
        internal TagCollection(Tag[] tags)
        {
            _tags = tags;
        }

        public IEnumerable<Tag> Tags => _tags;

        public Tag this[string key]
        {
            get
            {
                var tag = _tags.FirstOrDefault(x => x.Key == key);
                if (tag == null)
                    return new Tag(key, 0);
                
                return tag;
            }
        }       
    }
}