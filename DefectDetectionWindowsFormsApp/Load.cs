using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;

namespace DefectDetectionWindowsFormsApp
{
    public partial class Load : Form
    {
        //[StructLayoutAttribute(LayoutKind.Sequential)]
        //private struct DefectData
        //{
        //    public byte pinholeflag;
        //    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        //    public int[] box;    //矩形框
        //    public double ratio; //长宽比
        //    public double area; //面积
        //    public double avggray; //平均亮度
        //    public int classid; //瑕疵类别
        //    public int grade; //瑕疵等级
        //};
        private paramconf paramform;
        private Class1 class1;
        public Load()
        {
            InitializeComponent();
            paramform = new paramconf();
            class1 = new Class1();
            //string a = @"{""InspectModelName"":""249 - 4"",""FlawTemplateName"":""TERST"",""DetectDataTable"":[{""1"":""1"",""2"":""2"",""3"":""3"",""4"":""4""},{""1"":""250"",""2"":""250"",""3"":""249"",""4"":""249""}],""MaskDataTable"":[{""1"":""30"",""2"":""30"",""3"":""30""},{""1"":""2.54"",""2"":""2.54"",""3"":""2.54""}]}";
            string a = "";
            class1.SendDetectParam(a);
            class1.SendClassParam(a);
            //class1.SendJsonParam(a);
        }
        //[DllImport("D:\\C#\\DefectDetectionWindowsFormsApp\\Defect_detection.dll")]
        //static extern void handle(String imgpath, out IntPtr outputinfo, out int outputInfoLen);

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            MessageBox.Show("确认关闭页面");
        }


        private void button1_Click(object sender, EventArgs e)
        {
            paramform.Show();
        }

        private void button1_ClientSizeChanged(object sender, EventArgs e)
        {

        }

        private void button2_Click(object sender, EventArgs e)
        {
            int[] array = { 115, 140 };
            //class1.SendDetectParam(array, array.Length);
            string imgpath = "D:\\C#\\瑕疵检测资料\\test1.bmp";
            //string imgpath = "D:\\c++\\Defect_detection\\test.bmp";
            int outResultLen = -1;
            IntPtr outResultInfo;
            AvgPixelData outdata;
            Bitmap img = new Bitmap(imgpath);

            // Format8bppIndexed :像素值是颜色索引
            //ColorPalette tempPalette;
            //using (Bitmap tempBmp = new Bitmap(1, 1, PixelFormat.Format8bppIndexed))
            //{
            //    tempPalette = tempBmp.Palette;
            //}
            //for (int i = 0; i < 256; i++)
            //{
            //    tempPalette.Entries[i] = Color.FromArgb(i, i, i);
            //}
            //img.Palette = tempPalette;
            BitmapData imgData = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadWrite,
                PixelFormat.Format24bppRgb);

            class1.Detect(imgData.Scan0, imgData.Scan0, imgData.Width, imgData.Height, imgData.Stride, out outResultInfo, out outResultLen, out outdata);
            //handle(imgpath, out outResultInfo, out outResultLen);
            //Console.WriteLine("瑕疵个数是{0}", outResultLen);
            //List<DefectData> results = new List<DefectData>();
            //for (int i = 0; i < outResultLen; i++)
            //{
            //    results.Add((DefectData)Marshal.PtrToStructure((IntPtr)((outResultInfo + 
            //        i * Marshal.SizeOf(typeof(DefectData)))), typeof(DefectData)));

            //}
            //foreach (var seg in results)
            //{
            //    Console.WriteLine($"pinholeflag：{seg.pinholeflag}, box: [{string.Join(", ", seg.box)}], ratio: {seg.ratio},"+
            //        $"area: {seg.area}, avggray:{seg.avggray}");
            //}
            //string message = string.Format("序列  左x  左y  长度  宽度  长宽比  面积  平均亮度  \n");
            //for (int i=0;i<results.Count;i++)
            //{
            //    if (i % 8 == 0 && i != 0)
            //    {
            //        message = message + "\n";
            //    }
            //    string a = results[i].ToString()+"    ";
            //    message = message + a;

            //}

        }
    }
}
