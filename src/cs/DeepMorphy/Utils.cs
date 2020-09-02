using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Reflection;

namespace DeepMorphy
{
    internal static class Utils
    {
        private static Assembly _assembly = Assembly.GetExecutingAssembly();
        public static Stream GetResourceStream(string name)
        {
            return _assembly.GetManifestResourceStream(name);
        }

        public static Stream GetCompressedResourceStream(string name)
        {
            return new GZipStream(GetResourceStream(name), CompressionMode.Decompress);
        }
        
        public static IEnumerable<T> Yield<T>(this T item)
        {
            yield return item;
        }
    }
}