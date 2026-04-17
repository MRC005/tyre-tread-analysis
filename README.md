<h1 align="center">🚗 Tyre Tread Wear Severity Estimation</h1>

<p align="center">
<b>Classical Digital Image Processing Project (No Deep Learning)</b><br>
Low-cost tyre wear detection using smartphone images
</p>

<hr>

<h2>📌 Overview</h2>
<p>
This project estimates tyre tread condition (<b>Safe / Warning / Dangerous</b>) 
using <b>classical image processing and machine learning</b>.  
It avoids deep learning and focuses on an interpretable, lightweight pipeline 
suitable for real-world deployment on low-resource devices.
</p>

<hr>

<h2>🎯 Objective</h2>
<ul>
<li>Automatically detect worn-out tyres</li>
<li>Improve road safety through early detection</li>
<li>Provide a low-cost inspection system using smartphone images</li>
</ul>

<hr>

<h2>⚙️ System Pipeline</h2>

<h3>1️⃣ Preprocessing</h3>
<ul>
<li>Grayscale conversion</li>
<li>CLAHE (contrast enhancement)</li>
<li>Gaussian blur (noise reduction)</li>
</ul>

<h3>2️⃣ ROI Extraction</h3>
<ul>
<li>Canny edge detection</li>
<li>Morphological operations</li>
<li>Contour-based tread region extraction</li>
</ul>

<h3>3️⃣ Frequency Analysis</h3>
<ul>
<li>2D Discrete Fourier Transform (DFT)</li>
<li><b>TSCI (Tyre Surface Clarity Index)</b></li>
</ul>

<p><b>TSCI Formula:</b><br>
High-frequency energy / Total spectral energy</p>

<h3>4️⃣ Texture Analysis</h3>
<ul>
<li>GLCM features:
  <ul>
    <li>Contrast</li>
    <li>Dissimilarity</li>
    <li>Homogeneity</li>
    <li>Energy</li>
    <li>Correlation</li>
  </ul>
</li>
<li>Edge density (structural sharpness)</li>
</ul>

<h3>5️⃣ Classification</h3>
<ul>
<li>SVM (RBF kernel)</li>
<li>Standard scaling</li>
<li>Balanced class weighting</li>
</ul>

<hr>

<h2>📊 Results</h2>

<ul>
<li><b>Accuracy:</b> <b>74.8%</b></li>
<li><b>Worn Tyre Recall:</b> ~72%</li>
<li><b>Dataset Size:</b> 369 images</li>
<li><b>Validation:</b> 5-fold Stratified Cross Validation</li>
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
<td>~63.7%</td>
</tr>
<tr>
<td>GLCM only</td>
<td>~66.4%</td>
</tr>
<tr>
<td>Edge only</td>
<td>~64.0%</td>
</tr>
<tr>
<td>GLCM + Edge</td>
<td>~72.6%</td>
</tr>
<tr>
<td><b>Full (All Features)</b></td>
<td><b>74.8%</b></td>
</tr>
</table>

<p><b>Insight:</b> Texture features provide strong discrimination, while frequency and edge features improve robustness.</p>

<hr>

<h2>📁 Project Structure</h2>

<pre>
tyre-tread-project/
│
├── data/
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
│   ├── confusion_matrix.png
│   ├── ablation.png
│   ├── results.csv
│   └── results_final.csv
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
<li>Edge density improves structural detection</li>
<li>Feature fusion significantly improves accuracy</li>
<li>Dataset quality strongly impacts performance</li>
</ul>

<hr>

<h2>🚀 Applications</h2>
<ul>
<li>Roadside tyre inspection systems</li>
<li>Smartphone-based vehicle safety tools</li>
<li>Automated vehicle inspection pipelines</li>
</ul>

<hr>

<h2>⚠️ Limitations</h2>
<ul>
<li>Relatively small dataset (369 images)</li>
<li>Label noise affects performance</li>
<li>Sensitive to lighting and motion blur</li>
<li>No true multi-class ground truth</li>
</ul>

<hr>

<h2>📌 Author</h2>
<p><b>Machum Roy Choudhury</b></p>

<hr>

<h2>⭐ Note</h2>
<p>
This project is implemented entirely using <b>classical digital image processing</b> 
and machine learning techniques without deep learning.
</p>
