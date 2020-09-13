namespace DeepMorphy.Exception
{
    /// <summary>
    /// Выбрасывается при создании тега, если задана неподдерживаемая комбинация граммем
    /// </summary>
    public class TagNotFoundException : DeepMorphyException
    {
        public TagNotFoundException(string message) : base(message)
        {
            
        }
    }
}