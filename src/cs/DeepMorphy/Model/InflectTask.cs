namespace DeepMorphy.Model
{
    /// <summary>
    /// Задание для изменения формы слова
    /// </summary>
    public struct InflectTask
    {
        public InflectTask(string word, Tag wordTag, Tag resultTag)
        {
            this.word = word;
            this.wordTag = wordTag;
            this.resultTag = resultTag;
        }
        
        /// <summary>
        /// Слово, которое нужно поставить в другую форму
        /// </summary>
        public readonly string word;

        /// <summary>
        /// Тэг исходного слова
        /// </summary>
        public readonly Tag wordTag;

        /// <summary>
        /// Тэг, в который нужно поставить слово
        /// </summary>
        public readonly Tag resultTag;
    }
}