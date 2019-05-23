namespace DeepMorphy.PreProc
{
    interface IPreProcessor
    {
        MorphInfo Parse(string word);
    }
}