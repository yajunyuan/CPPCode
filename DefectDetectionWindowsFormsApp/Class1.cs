using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace DefectDetectionWindowsFormsApp
{
    [StructLayoutAttribute(LayoutKind.Sequential)]
    public struct AvgPixelData
    {
        public int avgpixelsize; //平均像素的长度
        public IntPtr avgpixel;  //每行平均像素的数组
    };
    class Class1
    {
        [StructLayoutAttribute(LayoutKind.Sequential)]
        private struct DefectData
        {
            public byte pinholeflag;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
            public int[] box;    //矩形框
            public double ratio; //长宽比
            public double area; //面积
            public double avggray; //平均亮度
            public int classid; //瑕疵类别
            public int grade; //瑕疵等级
        };

        [DllImport("D:\\C#\\DefectDetectionWindowsFormsApp\\Defect_detection.dll")]
        static extern void handle(IntPtr data, IntPtr data1, int width, int height, int stride, out IntPtr outputinfo, out int outputInfoLen, out AvgPixelData avgPixelData);
        //static extern void handle(String imgpath, out IntPtr outputinfo, out int outputInfoLen);
        [DllImport("D:\\C#\\DefectDetectionWindowsFormsApp\\Defect_detection.dll")]
        static extern void GetDetectJsonParam(String paramstr);
        [DllImport("D:\\C#\\DefectDetectionWindowsFormsApp\\Defect_detection.dll")]
        static extern void GetClassJsonParam(String paramstr);

        public void SendDetectParam(String paramstr)
        {
            GetDetectJsonParam(paramstr);
        }

        public void SendClassParam(String paramstr)
        {
            GetClassJsonParam(paramstr);
        }

        public void Detect(IntPtr data, IntPtr data1, int width, int height, int stride, out IntPtr outResultInfo, out int outResultLen, out AvgPixelData avgPixelData)
        {
            //string imgpath = "D:\\C#\\瑕疵检测资料\\清洗出拿来训练的图像\\pinjie\\pinjie-3.png";
            //int outResultLen = -1;
            //IntPtr outResultInfo;
            handle(data, data1, width, height, stride, out outResultInfo, out outResultLen, out avgPixelData);
            Console.WriteLine("瑕疵个数是{0}", outResultLen);
            List<DefectData> results = new List<DefectData>();
            for (int i = 0; i < outResultLen; i++)
            {
                results.Add((DefectData)Marshal.PtrToStructure((IntPtr)((outResultInfo +
                    i * Marshal.SizeOf(typeof(DefectData)))), typeof(DefectData)));

            }
            foreach (var seg in results)
            {
                Console.WriteLine($"pinholeflag：{seg.pinholeflag}, box: [{string.Join(", ", seg.box)}], ratio: {seg.ratio}," +
                    $"area: {seg.area}, avggray:{seg.avggray}");
            }
            byte[] avgpixeldata = new byte[avgPixelData.avgpixelsize];
            for (int j = 0; j < avgPixelData.avgpixelsize; j++)
            {
                avgpixeldata[j] = (byte)Marshal.PtrToStructure(avgPixelData.avgpixel + j * Marshal.SizeOf(typeof(byte)), typeof(byte));
                Console.WriteLine($"avgpixel: {avgpixeldata[j]}");
            }

        }

    }
}
