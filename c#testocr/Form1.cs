using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using OpenCvSharp;
namespace testocr
{
    [StructLayoutAttribute(LayoutKind.Sequential)]
    public struct OCRResult
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public int[] box;
        public int label;
        public int textlen;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 100)]
        public string text;
        //[MarshalAs(UnmanagedType.ByValArray, SizeConst = 100)]
        //public byte[] text;
        public double score;
        public bool flag;
    };
    public partial class Form1 : Form
    {
        [DllImport("DetectionInterface.dll")]
        static extern IntPtr AIInit();
        [DllImport("DetectionInterface.dll")]
        static extern void DetectInter(IntPtr h, IntPtr img, out int outputResultLen, out IntPtr outputResult, int detectmode);
        public Form1()
        {
            InitializeComponent();
            IntPtr init = AIInit();
            //Mat image = new Mat("C:\\Users\\rs\\Desktop\\ocr.png", ImreadModes.Color);
            Mat image = new Mat("D:\\c++\\tensorrt\\tensorrt\\test_x\\1-1.bmp", ImreadModes.Color);
            IntPtr imagePtr = image.CvPtr;
            int outLen = 1;
            IntPtr outputdata = IntPtr.Zero;
            int detectmode = 1;
            if (init == null&& detectmode==1)
            {
                Console.WriteLine("gpu detection is not supported");
                return;
            }
            DetectInter(init, imagePtr, out outLen, out outputdata, detectmode);
        
            OCRResult[] ads = new OCRResult[outLen];
            for (int i = 0; i < outLen; ++i)
            {
                ads[i] = (OCRResult)Marshal.PtrToStructure(outputdata + i * Marshal.SizeOf(typeof(OCRResult)),
                    typeof(OCRResult));
            }
            foreach (var seg in ads)
            {
                //Console.WriteLine(seg.text);
                Console.WriteLine($"text：{seg.text}, score: {seg.score}, flag: {seg.flag}," +
                    $"box: [{string.Join(", ", seg.box)}]");
                Console.WriteLine($"x: [{string.Join(", ", seg.text)}]");
                //string strGet = System.Text.Encoding.Default.GetString(seg.text, 0, seg.textlen);
                //Console.WriteLine(strGet);
            }
            
        }
    }
}
