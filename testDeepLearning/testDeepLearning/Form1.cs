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
using System.Drawing.Imaging;
using OpenCvSharp;

namespace testDeepLearning
{
    public partial class Form1 : Form
    {
        string modelfile;
        string mode;
        string imgfile;
        public Form1()
        {
            InitializeComponent();
        }
        [StructLayoutAttribute(LayoutKind.Sequential)]
        public struct RecResult
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 100)]
            public string imgname;
            public int reallabel;
            public int id;
            public double confidence;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
            public int[] box;
            public double radian;
            public IntPtr boxMask;
        };
        [DllImport("DeepLearning.dll")]
        static extern IntPtr AiGPUInit(String modelpath, String modelmode);
        [DllImport("DeepLearning.dll")]
        static extern void AIGPUDetectImg(IntPtr h, IntPtr data, int width, int height, int stride, out IntPtr result, out int outlen);
        private void button1_Click(object sender, EventArgs e)
        {
            getSelectedValue_Click();
            string qwq = @"D:\yolov8\ultralytics\runs\segment\train9\weights\best.engine";
            string qwq1 = "seg";
            IntPtr a = AiGPUInit(modelfile, mode);
            string qwq2 = @"D:\yolov8\ultralytics\data\segdata\images\val\2-36.bmp";
            Bitmap img = new Bitmap(imgfile);
            BitmapData imgData = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadWrite,
                PixelFormat.Format24bppRgb);
            int outLen = 0;
            IntPtr outputdata = IntPtr.Zero;
            int outmaskLen = 0;
            IntPtr ptr;
            AIGPUDetectImg(a, imgData.Scan0, imgData.Width, imgData.Height, imgData.Stride, out ptr, out outLen);
            // 将指针转换为结构体数组
            //for (int i = 0; i < n; i++)
            //{
            //    // 计算每个元素的偏移量
            //    IntPtr offset = IntPtr.Add(ptr, i * Marshal.SizeOf(typeof(OutputSeg)));
            //    // 从指针中读取结构体
            //    segs[i] = Marshal.PtrToStructure<OutputSeg>(offset);
            //}
            RecResult[] ads = new RecResult[outLen];
            for (int i = 0; i < outLen; ++i)
            {
                ads[i] = (RecResult)Marshal.PtrToStructure(ptr + i * Marshal.SizeOf(typeof(RecResult)), typeof(RecResult));
            }
            Mat image = new Mat(qwq2, ImreadModes.Color);
            foreach (var seg in ads)
            {
                Console.WriteLine($"id: {seg.id}, confidence: {seg.confidence}, box: [{string.Join(", ", seg.box)}], radian: {seg.radian}");
                //根据innerLen字段确定内部结构体数组的长度
                byte[,] segMasks = new byte[seg.box[3], seg.box[2]];
                //byte[] innerSegs = new byte[seg.box[2]* seg.box[3]];
                //for (int j = 0; j < seg.box[2] * seg.box[3]; j++)
                //{
                //    innerSegs[j] = (byte)Marshal.PtrToStructure(seg.boxMask + j * Marshal.SizeOf(typeof(byte)), typeof(byte));
                //}
                for (int i = 0; i < seg.box[3]; i++)
                {
                    for (int j = 0; j < seg.box[2]; j++)
                    {
                        segMasks[i, j] = (byte)Marshal.PtrToStructure(seg.boxMask + (i * seg.box[2] + j) * Marshal.SizeOf(typeof(byte)), typeof(byte));
                    }
                }
                Mat aa = new Mat(seg.box[3], seg.box[2], MatType.CV_8UC1, segMasks);
                Mat MyMat = new Mat(seg.box[3], seg.box[2], MatType.CV_8UC3, new Scalar(0, 0, 255));
                Mat ImageROI = new Mat(image, new Rect(seg.box[0], seg.box[1], seg.box[2], seg.box[3]));
                MyMat.CopyTo(ImageROI, aa);
            }
            Cv2.ImShow("chuli", image);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            // 创建打开文件对话框
            OpenFileDialog openFileDialog1 = new OpenFileDialog();

            // 设置对话框的属性
            //openFileDialog1.InitialDirectory = "C:\\";  // 设置默认文件夹
            openFileDialog1.Filter = "模型文件(onnx,engine)|*.onnx;*.engine";  // 设置文件类型过滤器

            // 显示打开文件对话框
            DialogResult result = openFileDialog1.ShowDialog();

            // 处理用户的文件选择
            if (result == DialogResult.OK)
            {
                // 获取用户所选文件的路径
                modelfile = openFileDialog1.FileName;
                button2.Text = modelfile;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            // 创建打开文件对话框
            OpenFileDialog openFileDialog1 = new OpenFileDialog();

            // 设置对话框的属性
            //openFileDialog1.InitialDirectory = "C:\\";  // 设置默认文件夹
            openFileDialog1.Filter = "图片文件(jpg,jpeg,bmp,gif,ico,pen,tif)|*.jpg;*.jpeg;*.bmp;*.gif;*.ico;*.png;*.tif;*.wmf";  // 设置文件类型过滤器

            // 显示打开文件对话框
            DialogResult result = openFileDialog1.ShowDialog();

            // 处理用户的文件选择
            if (result == DialogResult.OK)
            {
                // 获取用户所选文件的路径
                imgfile = openFileDialog1.FileName;
                button3.Text = imgfile;
            }
        }

        private void getSelectedValue_Click()
        {
            // 获取ComboBox中当前选定项的值
            mode = comboBox1.SelectedItem.ToString();

        }
    }
}
