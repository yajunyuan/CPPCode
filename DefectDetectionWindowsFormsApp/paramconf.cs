using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DefectDetectionWindowsFormsApp
{

    public partial class paramconf : Form
    {
        public int global_low_value;
        public paramconf()
        {
            InitializeComponent();
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            global_low_value = ((int)this.numericUpDown1.Value);
            MessageBox.Show(global_low_value.ToString());
        }

        private void button2_Click(object sender, EventArgs e)
        {
            MessageBox.Show(global_low_value.ToString());
        }

        private void paramconf_FormClosing(object sender, FormClosingEventArgs e)
        {
            this.Visible = false;
            e.Cancel = true;
        }
    }
}
