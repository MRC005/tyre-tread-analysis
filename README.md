<h1 align="center">🚗 Tyre Tread Wear Severity Estimation</h1>

<p align="center">
<b>Classical Digital Image Processing Project (No Deep Learning)</b><br>
Detect tyre wear from a single smartphone image
</p>

<hr>

<h2>📌 Overview</h2>
<p>
This project estimates tyre tread wear severity (<b>Safe / Warning / Dangerous</b>) 
using only <b>classical image processing techniques</b>.  
It avoids deep learning and focuses on a lightweight, interpretable pipeline.
</p>

<hr>

<h2>🎯 Objective</h2>
<ul>
<li>Detect worn-out tyres automatically</li>
<li>Improve road safety</li>
<li>Provide low-cost inspection using smartphone images</li>
</ul>

<hr>

<h2>⚙️ System Pipeline</h2>

<h3>1️⃣ Preprocessing (Module M2)</h3>
<ul>
<li>Grayscale conversion</li>
<li>CLAHE (Contrast enhancement)</li>
<li>Gaussian Blur (Noise removal)</li>
</ul>

<h3>2️⃣ ROI Extraction (Module M5)</h3>
<ul>
<li>Canny Edge Detection</li>
<li>Morphological Operations</li>
<li>Contour-based tread extraction</li>
</ul>

<h3>3️⃣ Frequency Analysis (Module M3)</h3>
<ul>
<li>2D Discrete Fourier Transform (DFT)</li>
<li><b>TSCI (Tyre Surface Clarity Index)</b> computation</li>
</ul>

<p><b>TSCI Formula:</b><br>
High Frequency Energy / Total Energy</p>

<h3>4️⃣ Texture Analysis (Module M6)</h3>
<ul>
<li>GLCM Features:
  <ul>
    <li>Contrast</li>
    <li>Dissimilarity</li>
    <li>Homogeneity</li>
    <li>Energy</li>
    <li>Correlation</li>
  </ul>
</li>
<li>LBP Histogram</li>
</ul>

<h3>5️⃣ Classification</h3>
<ul>
<li>SVM (RBF Kernel)</li>
<li>Balanced Class Weights</li>
</ul>

<hr>

<h2>📊 Results</h2>

<ul>
<li><b>Accuracy:</b> ~69%</li>
<li><b>Worn Tyre Recall:</b> ~70%</li>
<li><b>Improvement over baseline:</b> 32% → 69%</li>
</ul>

<hr>

<h2>📊 Ablation Study</h2>

<table border="1" cellpadding="6">
<tr>
<th>Feature Set</th>
<th>Accuracy</th>
</tr>
<tr>
<td>TSCI only</td>
<td>~63%</td>
</tr>
<tr>
<td>GLCM only</td>
<td>~66%</td>
</tr>
<tr>
<td><b>Combined (TSCI + GLCM)</b></td>
<td><b>~69%</b></td>
</tr>
</table>

<p><b>Insight:</b> GLCM provides better discrimination, while TSCI adds interpretability.</p>

<hr>

<h2>📁 Project Structure</h2>

<pre>
tyre-tread-project/
│
├── data/
│   ├── images/
│   ├── good/
│   └── bad/
│
├── src/
│   ├── stage1_preprocessing.py
│   ├── stage2_roi.py
│   ├── stage3_tsci.py
│   ├── stage4_texture.py
│   ├── stage5_fusion.py
│   ├── batch_process.py
│   ├── train_and_evaluate.py
│   └── main.py
│
├── outputs/
│   ├── output_stage1.png
│   ├── output_stage2.png
│   ├── output_stage3.png
│   ├── output_stage4.png
│   ├── confusion_matrix.png
│   ├── ablation.png
│   └── results.csv
│
└── requirements.txt
</pre>

<hr>

<h2>▶️ How to Run</h2>

<h3>Install Dependencies</h3>
<pre>pip install -r requirements.txt</pre>

<h3>Run Single Image</h3>
<pre>python src/main.py</pre>

<h3>Run Batch Processing</h3>
<pre>python src/batch_process.py</pre>

<h3>Train & Evaluate Model</h3>
<pre>python src/train_and_evaluate.py</pre>

<hr>

<h2>🧠 Key Insights</h2>
<ul>
<li>Frequency features capture groove sharpness</li>
<li>Texture features capture surface consistency</li>
<li>Combined approach improves performance</li>
<li>Dataset quality strongly affects results</li>
</ul>

<hr>

<h2>🚀 Applications</h2>
<ul>
<li>Roadside tyre inspection</li>
<li>Smartphone-based safety tools</li>
<li>Automated vehicle inspection systems</li>
</ul>

<hr>

<h2>⚠️ Limitations</h2>
<ul>
<li>Noisy dataset</li>
<li>No true 3-class ground truth</li>
<li>Sensitive to lighting and blur</li>
</ul>

<hr>

<h2>📌 Author</h2>
<p><b>Machum Roy Choudhury</b></p>

<hr>

<h2>⭐ Note</h2>
<p>
This project is implemented entirely using <b>classical digital image processing</b> 
without deep learning.
</p>
